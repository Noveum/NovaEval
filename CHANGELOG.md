# Changelog

All notable changes to NovaEval will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core framework
- Base classes for datasets, models, evaluators, and scorers
- Support for multiple model providers (OpenAI, Anthropic, AWS Bedrock)
- MMLU dataset implementation
- Accuracy and exact match scorers
- Configuration-based evaluation workflows
- CLI interface with comprehensive commands
- Docker containerization support
- Kubernetes deployment configurations
- Integration with Noveum.ai platform
- Comprehensive metrics calculation and reporting
- HTML, JSON, and CSV report generation
- Request logging and performance analytics
- Cost tracking and analysis
- S3 integration for artifact storage
- Extensible plugin architecture

### Features
- **Datasets**: MMLU, HuggingFace datasets, custom dataset support
- **Models**: OpenAI GPT models, Anthropic Claude, AWS Bedrock integration
- **Scorers**: Accuracy, exact match, F1 score, semantic similarity
- **Reporting**: Multiple output formats, detailed analytics, visualizations
- **Integrations**: Noveum.ai platform, AWS S3, credential management
- **Deployment**: Docker, Kubernetes, local development support

### Documentation
- Comprehensive README with examples
- API documentation and usage guides
- Contributing guidelines and development setup
- Docker and Kubernetes deployment guides
- Configuration reference and best practices

## [0.1.0] - 2024-07-12

### Added
- Initial release of NovaEval framework
- Core evaluation engine and plugin system
- Basic dataset and model support
- Command-line interface
- Docker containerization
- Documentation and examples

### Notes
- This is the initial release focusing on establishing the core framework
- Future releases will expand model support, add more datasets, and enhance reporting capabilities
- Integration with Noveum.ai platform provides seamless workflow management

---

## Release Planning

### Upcoming Features (v0.2.0)
- [ ] Additional dataset support (HellaSwag, TruthfulQA, GSM8K)
- [ ] More scoring mechanisms (BLEU, ROUGE, BERTScore)
- [ ] Enhanced visualization and reporting
- [ ] Batch evaluation optimization
- [ ] Advanced error handling and retry logic

### Future Roadmap (v0.3.0+)
- [ ] Multi-modal evaluation support
- [ ] Custom evaluation metrics
- [ ] Real-time evaluation monitoring
- [ ] Advanced analytics and insights
- [ ] Integration with more model providers
- [ ] Automated benchmark suites

### Long-term Vision
- [ ] AI agent evaluation capabilities
- [ ] Code generation benchmarks
- [ ] Conversational AI evaluation
- [ ] Safety and bias evaluation tools
- [ ] Enterprise features and scaling
