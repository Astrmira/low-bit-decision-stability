//! 在冻结QuIVer生产构图与查询入口采集真实访问轨迹。
use serde_json::{Value, json};
use std::{fs, io::{BufWriter, Write}, path::Path};
use triviumdb::index::{
    bq::{Bq2Signature, Bq2Store},
    quiver::{NavigationScorer, QuIVer, QuIVerConfig, QuIVerSearchConfig},
};
use triviumdb::vector::cosine_similarity_f32;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect("无法读取向量输入");
    assert_eq!(bytes.len() % 4, 0, "向量文件长度异常");
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn write_json(path: &Path, value: &Value) {
    let file = fs::File::create(path).expect("无法创建结果文件");
    serde_json::to_writer(BufWriter::new(file), value).expect("结果写入失败");
}

fn signatures(values: &[f32], dim: usize) -> Bq2Store {
    let mut store = Bq2Store::new(dim);
    store.reserve(values.len() / dim);
    for row in values.chunks_exact(dim) { store.push_from_vector(row); }
    store
}

struct TraceScorer<'a> {
    mode: &'a str,
    base: &'a [f32],
    query: &'a [f32],
    dim: usize,
    alternate: Option<&'a Bq2Store>,
    alternate_query: Bq2Signature,
    owner: &'a Bq2Store,
    owner_query: Bq2Signature,
    trace: Vec<(u32, u32, u32, u32)>,
}

impl NavigationScorer for TraceScorer<'_> {
    fn score(&mut self, id: u64, raw: u32) -> u32 {
        let internal = id as usize - 1;
        let full = self.owner.distance_to_sig(internal, &self.owner_query, self.dim);
        let score = match self.mode {
            "native" => raw,
            "full_sm" => full,
            "exact_l0" => {
                let row = &self.base[internal * self.dim..(internal + 1) * self.dim];
                let dot: f64 = row.iter().zip(self.query).map(|(&x, &y)| x as f64 * y as f64).sum();
                (1.0f32 - dot as f32).max(0.0).to_bits()
            }
            "cross" => self.alternate.unwrap().distance_to_sig(internal, &self.alternate_query, self.dim),
            _ => panic!("未知查询评分模式"),
        };
        self.trace.push((internal as u32, raw, score, full));
        score
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let config: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let out = Path::new(config["out"].as_str().unwrap());
    fs::create_dir_all(out)?;
    let dim = config["dim"].as_u64().unwrap() as usize;
    let base = read_f32(config["base"].as_str().unwrap());
    let n = base.len() / dim;
    assert_eq!(base.len(), n * dim);
    assert!(base.iter().all(|x| x.is_finite()), "输入含非有限数值");
    if args[1] == "build" {
        let mut graph = QuIVer::new(dim, &QuIVerConfig { m: 32, ef_construction: 128, alpha: 1.2 });
        let ids = (1..=n as u64).collect::<Vec<_>>();
        let slots = (0..n).collect::<Vec<_>>();
        let store = signatures(&base, dim);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        eprintln!("开始冻结生产路径构图，节点数{}，维度{}", n, dim);
        pool.install(|| graph.batch_build_with_store(&base, &ids, &slots, store));
        graph.save_to_file(&out.join("index.bin"))?;
        let mut adjacency = BufWriter::new(fs::File::create(out.join("graph.u32"))?);
        adjacency.write_all(&(n as u32).to_le_bytes())?;
        adjacency.write_all(&graph.ablation_entry_point().to_le_bytes())?;
        for node in 0..n {
            let neighbors = graph.layer0_neighbors(node as u32);
            adjacency.write_all(&(neighbors.len() as u32).to_le_bytes())?;
            for neighbor in neighbors { adjacency.write_all(&neighbor.to_le_bytes())?; }
        }
        adjacency.flush()?;
        write_json(&out.join("build.json"), &json!({
            "n": n, "dim": dim, "avg_degree": graph.stats().avg_degree_l0,
            "entry": graph.ablation_entry_point(), "config": config,
            "builder": "冻结QuIVer::batch_build_with_store生产路径",
        }));
        eprintln!("冻结生产路径构图完成");
        return Ok(());
    }
    assert!(args[1] == "query" || args[1] == "measure", "未知命令");
    let graph = QuIVer::load_from_file(Path::new(config["index"].as_str().unwrap()))?;
    let queries = read_f32(config["queries"].as_str().unwrap());
    if args[1] == "measure" {
        let evaluation_start = 128;
        let rounds = 5;
        let mut measurements = Vec::new();
        for width in config["widths"].as_array().unwrap() {
            let width = width.as_u64().unwrap() as usize;
            let search = QuIVerSearchConfig { top_k: 10, ef_search: width, rerank_limit: Some(64) };
            for query in queries.chunks_exact(dim).take(evaluation_start) {
                std::hint::black_box(graph.search_flat(query, &base, &search));
            }
            let mut observations = Vec::new();
            for round in 0..rounds {
                let mut labels = Vec::new();
                let mut latency_ns = Vec::new();
                let batch_start = std::time::Instant::now();
                for query in queries.chunks_exact(dim).skip(evaluation_start) {
                    let start = std::time::Instant::now();
                    let result = graph.search_flat(query, &base, &search);
                    latency_ns.push(start.elapsed().as_nanos() as u64);
                    labels.push(result);
                }
                let elapsed = batch_start.elapsed().as_secs_f64();
                observations.push(json!({"round":round,"elapsed_s":elapsed,"qps":labels.len() as f64/elapsed,"latency_ns":latency_ns,"labels":labels}));
            }
            measurements.push(json!({"ef":width,"observations":observations}));
            eprintln!("单线程五轮计量完成，宽度{}",width);
        }
        write_json(&out.join("measurement.json"), &json!({"evaluation_start":evaluation_start,"rounds":rounds,"measurements":measurements,"config":config,"timing":"仅原生公共search_flat查询，输入和索引加载、预热、序列化分别在计时边界外"}));
        return Ok(());
    }
    let modes = config["modes"].as_array().unwrap();
    let widths = config["widths"].as_array().unwrap();
    let alternate_vectors = read_f32(config["alternate_base"].as_str().unwrap());
    let alternate_queries = read_f32(config["alternate_queries"].as_str().unwrap());
    let alternate = signatures(&alternate_vectors, dim);
    let owner = signatures(&base, dim);
    assert_eq!(alternate_vectors.len(), base.len());
    assert_eq!(alternate_queries.len(), queries.len());
    for mode in modes {
        let mode = mode.as_str().unwrap();
        for width in widths {
            let width = width.as_u64().unwrap() as usize;
            let name = format!("{}-ef{}", mode, width);
            let mut raw = BufWriter::new(fs::File::create(out.join(format!("{}.trace.u32", name)))?);
            let mut results = Vec::new();
            for (qi, query) in queries.chunks_exact(dim).enumerate() {
                let mut scorer = TraceScorer {
                    mode, base: &base, query, dim, alternate: Some(&alternate),
                    alternate_query: Bq2Signature::from_vector(&alternate_queries[qi * dim..(qi + 1) * dim]),
                    owner: &owner, owner_query: Bq2Signature::from_vector(query),
                    trace: Vec::new(),
                };
                let mut rerank = Vec::new();
                let mut beam = Vec::new();
                let native = mode == "native";
                let mut labels = graph.search_with_scorer(query, |slot, buffer| {
                    beam.push(slot as u32);
                    if native {
                        buffer.clear(); buffer.extend_from_slice(&base[slot * dim..(slot + 1) * dim]);
                        rerank.push(slot as u32); true
                    } else {
                        // 只收集最终beam标签，返回false避免内核读取和精算整组候选。
                        false
                    }
                }, &QuIVerSearchConfig { top_k: 10, ef_search: width, rerank_limit: Some(if native {64} else {width}) }, &mut scorer);
                if !native {
                    // 各对照使用自己的评分选64个精排对象，避免内核最终BQ重排混入对照。
                    let mut score_by_node = vec![u32::MAX; n];
                    for &(id, _, chosen, _) in &scorer.trace { score_by_node[id as usize] = chosen; }
                    beam.sort_unstable_by_key(|&id| (score_by_node[id as usize], id));
                    rerank.extend(beam.iter().take(64).copied());
                    labels = rerank.iter().map(|&id| {
                        (id as u64 + 1, cosine_similarity_f32(query, &base[id as usize * dim..(id as usize + 1) * dim]))
                    }).collect();
                    labels.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    labels.truncate(10);
                }
                // 采集器对照原生公共入口，确认记录访问不会改变查询输出。
                if mode == "native" && qi < 8 {
                    let expected = graph.search_flat(query, &base, &QuIVerSearchConfig { top_k: 10, ef_search: width, rerank_limit: Some(64) });
                    assert_eq!(labels.iter().map(|r| r.0).collect::<Vec<_>>(), expected.iter().map(|r| r.0).collect::<Vec<_>>(), "记录器改变了公共入口输出");
                }
                raw.write_all(&(qi as u32).to_le_bytes())?;
                raw.write_all(&(scorer.trace.len() as u32).to_le_bytes())?;
                for &(id, bq, chosen, full) in &scorer.trace {
                    for value in [id, bq, chosen, full] { raw.write_all(&value.to_le_bytes())?; }
                }
                results.push(json!({"query":qi,"labels":labels,"scored":scorer.trace.len(),"rerank":rerank}));
            }
            raw.flush()?;
            write_json(&out.join(format!("{}.json",name)), &json!({"mode":mode,"ef":width,"queries":results,"config":config,"trace_columns":["id","cheap","chosen","full_sm"],"scope":"native完整冻结生产查询；其余模式共享该图的高层入口遍历，更换L0评分并按相应评分选64个精排对象"}));
            eprintln!("查询轨迹完成，模式{}，宽度{}", mode, width);
        }
    }
    Ok(())
}
