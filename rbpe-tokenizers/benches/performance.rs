//! Performance benchmarks for R-BPE tokenizer

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rbpe_tokenizers::model::RBPEModel;
use rbpe_tokenizers::pretokenizer::RBPEPreTokenizer;
use rbpe_tokenizers::utils::UnicodeRangeChecker;
use rbpe_tokenizers::utils::unicode_ranges::ranges;
use std::path::Path;

fn create_model() -> RBPEModel {
    let base_path = Path::new("../rbpe_tokenizer");
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    RBPEModel::from_files(
        &base_path.join("new_tokenizer/tokenizer.json"),
        &base_path.join("old_tokenizer/tokenizer.json"),
        &base_path.join("metadata/new_to_old_map.json"),
        &base_path.join("metadata/old_to_new_map.json"),
        Some(&base_path.join("metadata/replacement_character_map.json")),
        pretokenizer,
        None,
    ).expect("Failed to load model")
}

fn bench_encode(c: &mut Criterion) {
    let model = create_model();
    
    let medium_english = "The quick brown fox jumps over the lazy dog. ".repeat(5);
    let medium_arabic = "مرحبا! كيف حالك اليوم؟ ".repeat(5);
    let medium_mixed = "This is mixed text هذا نص مختلط ".repeat(5);
    let long_english = "This is a longer piece of text that contains multiple sentences. ".repeat(20);
    let long_arabic = "هذا نص طويل يحتوي على جمل متعددة. ".repeat(20);
    let long_mixed = "This is mixed text هذا نص مختلط ".repeat(20);
    
    let test_cases = vec![
        ("short_english", "Hello World"),
        ("short_arabic", "مرحبا"),
        ("mixed", "Hello مرحبا World"),
        ("medium_english", medium_english.as_str()),
        ("medium_arabic", medium_arabic.as_str()),
        ("medium_mixed", medium_mixed.as_str()),
        ("long_english", long_english.as_str()),
        ("long_arabic", long_arabic.as_str()),
        ("long_mixed", long_mixed.as_str()),
    ];
    
    let mut group = c.benchmark_group("encode");
    
    for (name, text) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("encode", name),
            &text,
            |b, text| {
                b.iter(|| {
                    model.encode(black_box(text), false).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let model = create_model();
    
    let medium_english = "The quick brown fox jumps over the lazy dog. ".repeat(5);
    let medium_arabic = "مرحبا! كيف حالك اليوم؟ ".repeat(5);
    let long_english = "This is a longer piece of text that contains multiple sentences. ".repeat(20);
    
    // Pre-encode test cases
    let test_cases = vec![
        ("short_english", model.encode("Hello World", false).unwrap()),
        ("short_arabic", model.encode("مرحبا", false).unwrap()),
        ("mixed", model.encode("Hello مرحبا World", false).unwrap()),
        ("medium_english", model.encode(&medium_english, false).unwrap()),
        ("medium_arabic", model.encode(&medium_arabic, false).unwrap()),
        ("long_english", model.encode(&long_english, false).unwrap()),
    ];
    
    let mut group = c.benchmark_group("decode");
    
    for (name, ids) in test_cases {
        // Basic decode
        group.bench_with_input(
            BenchmarkId::new("basic_decode", name),
            &ids,
            |b, ids| {
                b.iter(|| {
                    model.decode(black_box(ids), false).unwrap()
                });
            },
        );
        
        // Advanced decode
        group.bench_with_input(
            BenchmarkId::new("advanced_decode", name),
            &ids,
            |b, ids| {
                b.iter(|| {
                    model.decode_advanced(black_box(ids), false).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let model = create_model();
    
    let medium = "This is mixed text هذا نص مختلط ".repeat(5);
    let long = "This is a longer piece of text هذا نص أطول ".repeat(20);
    
    let test_cases = vec![
        ("short", "Hello مرحبا World"),
        ("medium", medium.as_str()),
        ("long", long.as_str()),
    ];
    
    let mut group = c.benchmark_group("roundtrip");
    
    for (name, text) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("encode_decode", name),
            &text,
            |b, text| {
                b.iter(|| {
                    let ids = model.encode(black_box(text), false).unwrap();
                    model.decode(black_box(&ids), false).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pretokenizer(c: &mut Criterion) {
    let checker = UnicodeRangeChecker::new(ranges::all_arabic());
    let pretokenizer = RBPEPreTokenizer::new(checker, vec![]);
    
    let medium = "This is mixed text هذا نص مختلط ".repeat(5);
    let long = "This is a longer piece of text هذا نص أطول ".repeat(20);
    let very_long = "Mixed language content محتوى متعدد اللغات ".repeat(100);
    
    let test_cases = vec![
        ("short", "Hello مرحبا World"),
        ("medium", medium.as_str()),
        ("long", long.as_str()),
        ("very_long", very_long.as_str()),
    ];
    
    let mut group = c.benchmark_group("pretokenizer");
    
    for (name, text) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("segment", name),
            &text,
            |b, text| {
                b.iter(|| {
                    pretokenizer.pre_tokenize(black_box(text))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_roundtrip,
    bench_pretokenizer
);
criterion_main!(benches);
