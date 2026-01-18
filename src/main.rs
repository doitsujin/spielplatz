use std::env;
use std::mem;
use std::path::{Path};

use clap::{Parser};

mod batch;
mod data_conv;
mod shader;
mod shader_reflection;
mod vulkan;

use batch::*;
use vulkan::*;

#[derive(Parser)]
#[command(version)]
struct CliArgs {
    // Batch file
    json : Option<String>,

    #[arg(long, value_name = "name")]
    batch : Option<String>,

    #[arg(long, value_name = "output")]
    display : Option<String>,

    #[arg(long, value_name = "index")]
    adapter : Option<u32>,

    #[arg(long)]
    debug : bool,
}

fn run_batch_mode(
    args        : &CliArgs,
    mut vulkan  : Context,
    batch_file  : BatchFile,
    base_path   : &Path
) -> Result<(), String> {
    let shaders = batch_file.load_shaders(&base_path, &vulkan)?;

    let mut found_batch = false;

    for batch in batch_file.list_batches() {
        let skip_batch = args.batch.as_ref().map(
            |name| name != &batch).unwrap_or(false);

        if skip_batch {
            continue;
        }

        let instance = batch_file.load_batch(&base_path,
            &mut vulkan, &shaders, &batch)?;

        instance.dispatch(&mut vulkan)?;
        instance.store_outputs(&mut vulkan)?;

        found_batch = true;
    }

    if !found_batch {
        return Err(format!("No batch named '{}' found",
            args.batch.clone().unwrap_or_default()));
    }

    Ok(())
}

fn run_interactive_mode(
    args        : &CliArgs,
    mut vulkan  : Context,
    batch_file  : BatchFile,
    base_path   : &Path,
    output      : &str
) -> Result<(), String> {
    let shaders = batch_file.load_shaders(&base_path, &vulkan)?;

    let mut batch_index = 0usize;
    let batches = batch_file.list_batches();

    if batches.is_empty() {
        return Err("No batches in file.".to_string());
    }

    let mut instance_container = None::<BatchInstance>;

    loop {
        if instance_container.is_none() {
            let batch_name = &batches[batch_index];
            println!("Loading batch {batch_name}");

            instance_container = Some(batch_file.load_batch(&base_path,
                &mut vulkan, &shaders, batch_name)?);
        }

        let instance = instance_container.as_ref().unwrap();

        let Some(display_image) = instance.get_output_image(output) else {
            return Err(format!("Resource {output} not found or not an image"));
        };

        instance.dispatch(&mut vulkan)?;

        match vulkan.display(&display_image)? {
            Action::Quit => { break; },

            Action::NextBatch => {
                mem::drop(instance_container.take());
                batch_index = (batch_index + 1) % batches.len();
            },

            Action::PrevBatch => {
                mem::drop(instance_container.take());
                batch_index = (batch_index + batches.len() - 1) % batches.len();
            },

            _ => { },
        }
    }

    Ok(())
}

fn run_app() -> Result<(), String> {
    let args = CliArgs::parse();

    let json_path = Path::new(args.json.as_ref().ok_or("No JSON file specified".to_string())?);
    let batch_file = BatchFile::parse_file(&json_path)?;

    let mut vulkan = Context::new(VulkanInfo::new()
        .adapter_index(args.adapter.unwrap_or(0))
        .debug(args.debug))?;

    let base_path = json_path.parent().ok_or("Invalid path".to_string())?;

    if let Some(output) = &args.display {
        run_interactive_mode(&args, vulkan, batch_file, base_path, output)
    } else {
        run_batch_mode(&args, vulkan, batch_file, base_path)
    }
}

fn main() {
    if let Err(e) = run_app() {
        eprintln!("{e}");
    }
}
