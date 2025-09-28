use std::{env::{self, Args}, io::Write};

use embeddings::Embeddings;
use ::llm::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use output_projection::OutputProjection;
use transformer::TransformerBlock;
use llm::LLM;
use vocab::Vocab;
use rand::SeedableRng;
use rand::rngs::StdRng;
// use crate::RAND_SEED;
// StdRng::seed_from_u64(RAND_SEED); // Use any u64 value as seed

mod dataset_loader;
mod llm;
mod embeddings;
mod vocab;
mod transformer;
mod feed_forward;
mod self_attention;
mod output_projection;
mod adam;
mod layer_norm;

const NUM_EPOCHS: usize = 68;
const LR: f32 = 0.0005;

const RAND_SEED: u64 = 0;
use rand_distr::{Distribution, Normal};

fn main() {
    let mut args: Args = env::args();

    args.next();

    let num_epochs = args.next();

    let num_epochs = num_epochs
        .map(|str| str.parse().unwrap_or(NUM_EPOCHS));

    let num_epochs = num_epochs.unwrap_or(NUM_EPOCHS);

    let lr = args.next()
        .map(|x| x.parse().unwrap_or(LR))
        .unwrap_or(LR);
    let mut rng = StdRng::seed_from_u64(RAND_SEED);  // rand::rng()
    let normal = Normal::new(0.0, 1.0).unwrap();
    println!("First normal rng(seed={}) sample: {}", RAND_SEED, normal.sample(&mut rng));
 
    // Mock input - test conversational format
    let string = String::from("User: How do mountains form?");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();
    
    // Add end of sequence token
    vocab_set.insert("</s>".to_string());
    
    
    let dataset = dataset_loader::Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        dataset_loader::DatasetType::JSON,
    ); // Placeholder, not used in this example

    // Process all training examples for vocabulary
    // First process pre-training data
    for text in &dataset.pretraining_data {
        for word in text.split_whitespace() {
            // Handle punctuation by splitting it from words
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        vocab_set.insert(current.clone());
                        current.clear();
                    }
                    vocab_set.insert(c.to_string());
                } else {
                    current.push(c);
                }
            }
            if !current.is_empty() {
                vocab_set.insert(current);
            }
        }
    }
    
    // Then process chat training data
    for row in &dataset.chat_training_data {
        // Add words from outputs
        for word in row.split_whitespace() {
            // Handle punctuation by splitting it from words
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        vocab_set.insert(current.clone()); // Clone to avoid moving
                        current.clear(); // Use clear() instead of String::new()
                    }
                    vocab_set.insert(c.to_string());
                } else {
                    current.push(c);
                }
            }
            if !current.is_empty() {
                vocab_set.insert(current);
            }
        }
    }
    
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort(); // Sort for deterministic ordering
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s: &String| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    // let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    // let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    let mut llm = LLM::new(vocab, vec![
        Box::new(embeddings),
        Box::new(transformer_block_1),
        // Box::new(transformer_block_2),
        // Box::new(transformer_block_3),
        Box::new(output_projection),
    ]);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    
    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));
    
    println!("\n=== PRE-TRAINING MODEL ===");
    println!("Pre-training on {} examples for {} epochs with learning rate {}", 
             dataset.pretraining_data.len(), num_epochs, lr);
    llm.train(
        dataset.pretraining_data.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
        num_epochs,
        lr
    );
    
    println!("\n=== INSTRUCTION TUNING ===");
    println!("Instruction tuning on {} examples for {} epochs with learning rate {}", 
             dataset.chat_training_data.len(), num_epochs, lr / 5.0);
    llm.train(
        dataset.chat_training_data.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
        num_epochs, 
        lr / 5.0
    ); // Much lower learning rate for stability

 
    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");
    
    let mut input = String::new();
    loop {
        // Clear the input string
        input.clear();
        
        // Prompt for user input
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();
        
        // Read user input
        std::io::stdin().read_line(&mut input).expect("Failed to read input");
        
        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }
        
        // Generate prediction based on user input with "User:" prefix
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }
}
