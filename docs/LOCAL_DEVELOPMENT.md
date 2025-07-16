---
layout: default
title: Local Development
nav_order: 8
---

# Local GitHub Pages Development Guide

This guide helps you build and test the NovaEval documentation locally using Jekyll.

## 🚀 Prerequisites

1. Install Ruby

⚠️ **Important**: Jekyll 3.10.0 (required for GitHub Pages compatibility) requires Ruby 3.2 or lower. Newer Ruby versions (≥3.3) will cause native-extension build errors with gems like psych and nokogiri.

**Recommended: Use a Ruby Version Manager**

Using a Ruby version manager makes it easy to switch between Ruby versions:

**Install rbenv (recommended):**

```bash
# macOS
brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Ubuntu/Debian
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Ruby 3.2.x
rbenv install 3.2.2
rbenv global 3.2.2
```

**Alternative: Install asdf:**
```bash
# macOS
brew install asdf
echo '. $(brew --prefix asdf)/libexec/asdf.sh' >> ~/.zshrc
source ~/.zshrc

# Ubuntu/Debian
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.13.1
echo '. "$HOME/.asdf/asdf.sh"' >> ~/.bashrc
source ~/.bashrc

# Install Ruby plugin and Ruby 3.2.x
asdf plugin add ruby
asdf install ruby 3.2.2
asdf global ruby 3.2.2
```

**Direct Installation (if you must):**

**macOS:**
```bash
# Install specific Ruby version via Homebrew
brew install ruby@3.2
echo 'export PATH="/opt/homebrew/opt/ruby@3.2/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Ubuntu/Debian:**
```bash
# Install Ruby 3.2 via apt (if available)
sudo apt-get update
sudo apt-get install ruby-full build-essential zlib1g-dev

# Note: This may install Ruby 3.3+, which is incompatible
# Use rbenv or asdf for better control
```

**Windows:**
- Install [Ruby+Devkit 3.2.x](https://rubyinstaller.org/downloads/) (avoid 3.3+)
- Use the RubyInstaller for Windows

### 2. Install Bundler

```bash
gem install bundler
```

## 🔧 Setup

### 1. Navigate to the docs directory

```bash
cd docs
```

### 2. Install dependencies

```bash
bundle install
```

This will install all the required gems including Jekyll 3.10.0 (via github-pages), GitHub Pages, and all plugins.

### 3. Build and serve the site

```bash
# Build and serve (development mode)
bundle exec jekyll serve

# Or with live reload (automatically refreshes browser)
bundle exec jekyll serve --livereload

# Serve on a specific port
bundle exec jekyll serve --port 4001

# Build for production
bundle exec jekyll build
```

### 4. Access your site

Open your browser and go to:
- **Local development**: http://localhost:4000/NovaEval/
- **Custom port**: http://localhost:4001/NovaEval/

## 📋 Common Commands

### Development
```bash
# Start development server
bundle exec jekyll serve --livereload

# Build site (output to _site/)
bundle exec jekyll build

# Build for production (with environment)
JEKYLL_ENV=production bundle exec jekyll build

# Clean build files
bundle exec jekyll clean
```

### Testing
```bash
# Check for broken links (install html-proofer first)
gem install html-proofer
bundle exec jekyll build
bundle exec htmlproofer ./_site
```

### Updating Dependencies
```bash
# Update gems
bundle update

# Update to latest GitHub Pages version
bundle update github-pages
```

## 🔍 Troubleshooting

### Common Issues

**1. Ruby version conflicts**
```bash
# Check Ruby version
ruby --version

# Should be 3.2.x or lower (NOT 3.3+)
# Jekyll 3.10.0 is incompatible with Ruby 3.3+
```

**If you have Ruby 3.3+, downgrade using a version manager:**
```bash
# Using rbenv
rbenv install 3.2.2
rbenv global 3.2.2

# Using asdf
asdf install ruby 3.2.2
asdf global ruby 3.2.2

# Verify the change
ruby --version
```

**2. Native extension build errors**
```bash
# If you see errors like:
# - "psych" gem failed to build
# - "nokogiri" gem failed to build
# - "ffi" gem failed to build

# This usually means Ruby version is too new (≥3.3)
# Downgrade to Ruby 3.2.x using the commands above
```

**3. Permission errors on macOS**
```bash
# If you get permission errors, use:
sudo gem install bundler

# Or better, use rbenv/asdf for Ruby version management
# This avoids system Ruby conflicts entirely
```

**4. Port already in use**
```bash
# Use a different port
bundle exec jekyll serve --port 4001
```

**5. Bundler::GemNotFound errors**
```bash
# Clean and reinstall
bundle clean --force
bundle install
```

### Performance Tips

**1. Incremental builds**
```bash
bundle exec jekyll serve --incremental
```

**2. Skip regeneration of CSS/JS**
```bash
bundle exec jekyll serve --skip-initial-build
```

**3. Limit posts (if you have many)**
```bash
bundle exec jekyll serve --limit_posts 10
```

## 📝 Development Workflow

### 1. Making Changes

1. Edit markdown files in the `docs/` directory
2. The site will automatically rebuild (if using `--livereload`)
3. Refresh your browser to see changes

### 2. Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add proper front matter:
   ```yaml
   ---
   title: Your Page Title
   description: Page description
   ---
   ```

### 3. Testing Before Deployment

```bash
# Test the production build
JEKYLL_ENV=production bundle exec jekyll build
bundle exec jekyll serve --no-watch
```

## 🌐 GitHub Pages Simulation

**Important**: We use Jekyll 3.10.0 (via the `github-pages` gem) instead of Jekyll 4.x to ensure 100% compatibility with GitHub Pages hosting environment. This prevents deployment issues and ensures your local development exactly matches the production environment.

**⚠️ Ruby Version Compatibility**: Jekyll 3.10.0 requires Ruby 3.2 or lower. If you encounter native-extension build errors (psych, nokogiri, etc.), your Ruby version is likely too new (≥3.3). Use a Ruby version manager like rbenv or asdf to downgrade to Ruby 3.2.x.

To exactly match GitHub Pages environment:

### 1. Use GitHub Pages gem
```bash
# This is already in the Gemfile
bundle exec jekyll serve
```

### 2. Check GitHub Pages versions
```bash
# List GitHub Pages dependencies
bundle exec github-pages versions
```

### 3. Test with GitHub Pages health check
```bash
# Install the gem
gem install github-pages-health-check

# Run the check
github-pages-health-check
```

## 🚦 Continuous Testing

### GitHub Actions Workflow
The `.github/workflows/docs.yml` file automatically:
- Builds the site on every push
- Deploys to GitHub Pages
- Runs tests

### Pre-deployment Checks
Before pushing changes:

```bash
# Build and test locally
bundle exec jekyll build
bundle exec htmlproofer ./_site --check-html --check-opengraph

# Check links
bundle exec jekyll build
bundle exec htmlproofer ./_site --check-external-links
```

## 🔧 Advanced Configuration

### Custom Configuration for Local Development

Create `docs/_config_dev.yml`:
```yaml
# Development overrides
url: http://localhost:4000
baseurl: /NovaEval
environment: development

# Disable analytics in development
google_analytics: false
```

Use it with:
```bash
bundle exec jekyll serve --config _config.yml,_config_dev.yml
```

### Environment-Specific Settings

```bash
# Development
JEKYLL_ENV=development bundle exec jekyll serve

# Production
JEKYLL_ENV=production bundle exec jekyll build
```

## 📚 Useful Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Minima Theme](https://github.com/jekyll/minima)
- [Jekyll on macOS](https://jekyllrb.com/docs/installation/macos/)

## 🆘 Getting Help

If you encounter issues:

1. Check the [Jekyll troubleshooting guide](https://jekyllrb.com/docs/troubleshooting/)
2. Look at the [GitHub Pages documentation](https://docs.github.com/en/pages)
3. Check our [GitHub Issues](https://github.com/Noveum/NovaEval/issues)

---

**Happy documenting!** 📚✨
