# NovaEval Documentation

This directory contains the documentation for NovaEval, built with Jekyll and hosted on GitHub Pages.

## 📖 Documentation Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── Gemfile             # Ruby dependencies
├── index.md            # Homepage
├── getting-started.md  # Getting started guide
├── user-guide.md       # Comprehensive user guide
├── examples.md         # Real-world examples
├── api-reference.md    # API documentation
├── contributing.md     # Contributing guide
└── README.md          # This file
```

## 🚀 Running Locally

### Prerequisites

- **Ruby 3.0+** installed
- **Bundler** gem installed

### Setup

1. **Install dependencies**:
   ```bash
   cd docs
   bundle install
   ```

2. **Serve locally**:
   ```bash
   bundle exec jekyll serve
   ```

3. **View the site**:
   Open http://localhost:4000 in your browser

### Development Commands

```bash
# Install dependencies
bundle install

# Serve with live reload
bundle exec jekyll serve --livereload

# Serve with drafts
bundle exec jekyll serve --drafts

# Build for production
bundle exec jekyll build

# Clean build files
bundle exec jekyll clean
```

## 📝 Writing Documentation

### Markdown Files

All documentation is written in Markdown with Jekyll front matter:

```markdown
---
layout: default
title: Page Title
nav_order: 1
---

# Page Title

Your content here...
```

### Front Matter Options

- **layout**: Page layout (default: `default`)
- **title**: Page title for navigation and SEO
- **nav_order**: Order in navigation menu
- **description**: Page description for SEO
- **permalink**: Custom URL path

### Code Blocks

Use syntax highlighting for code examples:

````markdown
```python
from novaeval import Evaluator

evaluator = Evaluator(dataset, models, scorers)
results = evaluator.run()
```
````

### Links

Use relative links for internal pages:

```markdown
[Getting Started](getting-started.md)
[API Reference](api-reference.md)
```

### Images

Store images in an `assets/images/` directory:

```markdown
![NovaEval Logo](assets/images/logo.png)
```

## 🎨 Styling

### Custom CSS

Add custom styles in the Jekyll front matter:

```markdown
<style>
.custom-class {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
}
</style>
```

### Components

Use HTML for custom components:

```html
<div class="alert alert-info">
  <strong>Note:</strong> This is an informational alert.
</div>
```

## 🔧 Configuration

### Jekyll Configuration

Key settings in `_config.yml`:

```yaml
title: NovaEval Documentation
description: AI model evaluation framework
url: https://noveum.github.io/NovaEval
baseurl: /NovaEval

# Theme and plugins
theme: minima
plugins:
  - jekyll-feed
  - jekyll-sitemap
  - jekyll-seo-tag
```

### Navigation

Navigation is configured in `_config.yml`:

```yaml
nav_links:
  - title: Home
    url: /
  - title: Getting Started
    url: /getting-started/
  - title: User Guide
    url: /user-guide/
```

## 🚀 Deployment

### GitHub Pages

Documentation is automatically deployed to GitHub Pages when:

1. **Push to main branch** with changes in `docs/` directory
2. **Manual workflow dispatch** from GitHub Actions

### Custom Domain

To use a custom domain:

1. **Add CNAME file** to `docs/` directory
2. **Configure domain** in repository settings
3. **Update `_config.yml`** with new URL

## 📚 Best Practices

### Writing Style

- **Use clear, concise language**
- **Include practical examples**
- **Add code comments** for complex examples
- **Link to related sections**
- **Keep content up to date**

### Structure

- **Start with overview** and key concepts
- **Provide step-by-step instructions**
- **Include troubleshooting sections**
- **Add cross-references** between pages

### Code Examples

- **Use realistic examples** from actual use cases
- **Include complete, runnable code**
- **Add explanatory comments**
- **Show expected output**
- **Test examples** before publishing

### Maintenance

- **Regular reviews** for accuracy
- **Update examples** with new features
- **Fix broken links**
- **Improve based on user feedback**

## 🤝 Contributing to Documentation

### Process

1. **Fork the repository**
2. **Create a branch** for your changes
3. **Edit documentation** files
4. **Test locally** with Jekyll
5. **Submit a pull request**

### Guidelines

- **Follow the writing style** of existing documentation
- **Add examples** for new features
- **Update navigation** if adding new pages
- **Test all links** and code examples
- **Preview changes** locally before submitting

### Review Process

Documentation changes are reviewed for:

- **Technical accuracy**
- **Writing quality**
- **Consistency** with existing docs
- **Completeness** of examples
- **SEO optimization**

## 📞 Support

Need help with documentation?

- **GitHub Issues**: [Report documentation issues](https://github.com/Noveum/NovaEval/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Noveum/NovaEval/discussions)
- **Email**: [team@noveum.ai](mailto:team@noveum.ai)

## 📄 License

Documentation is licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

**Built with ❤️ by the [Noveum.ai](https://noveum.ai) team**
