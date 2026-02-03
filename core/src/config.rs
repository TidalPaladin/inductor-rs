//! Configuration types for inductor-rs.

use serde::Deserialize;

/// Top-level configuration.
#[derive(Debug, Default, Deserialize)]
pub struct Config {
    /// Model configuration.
    #[serde(default)]
    pub model: ModelConfig,

    /// Inference configuration.
    #[serde(default)]
    pub inference: InferenceConfig,
}

/// Model configuration.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Path to the .pt2 model package.
    #[serde(default)]
    pub package_path: Option<String>,

    /// Device to load model on.
    #[serde(default = "default_device")]
    pub device: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            package_path: None,
            device: default_device(),
        }
    }
}

/// Inference configuration.
#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
        }
    }
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_batch_size() -> usize {
    1
}

impl Config {
    /// Load configuration from a YAML file.
    pub fn from_yaml_file(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from a YAML string.
    pub fn from_yaml_str(yaml: &str) -> crate::error::Result<Self> {
        let config: Config = serde_yaml::from_str(yaml)?;
        Ok(config)
    }
}
