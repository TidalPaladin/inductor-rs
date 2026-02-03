//! Error types for inductor-rs.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for inductor-rs operations.
pub type Result<T> = std::result::Result<T, InductorError>;

/// Errors that can occur during model inference.
#[derive(Debug, Error)]
pub enum InductorError {
    /// Model loading failed.
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    /// Inference failed.
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Invalid tensor.
    #[error("Invalid tensor: {0}")]
    Tensor(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON parsing error.
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// File not found.
    #[error("File not found: {}", .0.display())]
    FileNotFound(PathBuf),
}

impl InductorError {
    /// Create a model load error.
    pub fn model_load(msg: impl Into<String>) -> Self {
        Self::ModelLoad(msg.into())
    }

    /// Create an inference error.
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a tensor error.
    pub fn tensor(msg: impl Into<String>) -> Self {
        Self::Tensor(msg.into())
    }

    /// Create a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = InductorError::model_load("failed to load");
        assert_eq!(format!("{}", err), "Model loading failed: failed to load");

        let err = InductorError::config("invalid dtype");
        assert_eq!(format!("{}", err), "Configuration error: invalid dtype");

        let err = InductorError::FileNotFound(PathBuf::from("/path/to/model.pt2"));
        assert_eq!(format!("{}", err), "File not found: /path/to/model.pt2");
    }
}
