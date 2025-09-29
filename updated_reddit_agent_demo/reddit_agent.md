# Reddit Agent Execution Flow and Tracing Documentation

## Overview

The Reddit Agent is an automated system that finds relevant Reddit discussions for APIs and generates contextual comments to promote those APIs. The system uses comprehensive tracing via the Noveum Trace SDK to monitor and debug the entire execution process.

## System Architecture

The agent consists of several key components:
- **Main Executor** (`agent_executor.py`): Orchestrates the entire workflow
- **API Management**: Sitemap parsing, API selection, and metadata extraction
- **Search & Discovery**: Query generation and Reddit post discovery via Tavily
- **Content Generation**: AI-powered comment generation using Gemini
- **State Management**: Tracking used APIs and subreddits to avoid repetition
- **Output Generation**: HTML report creation and email delivery

## Execution Flow

### 1. System Initialization
- Loads environment variables (API keys, email settings)
- Initializes Noveum Trace SDK with project configuration
- Sets up target parameters: 2 successful runs, 2 max attempts per run

### 2. Main Execution Loop
The system runs in a while loop until it completes 2 successful runs:

#### 2.1 Run Initialization
- Creates a new trace span for each run (`reddit_agent_run_{run_number}`)
- Sets comprehensive run-level attributes (target runs, max attempts, timing)
- Initializes inner attempt counter

#### 2.2 Inner Attempt Loop
For each run, the system attempts up to 2 times to find valid Reddit posts:

### 3. API Selection Phase
**Traced Operation**: `api_selection`

- **Sitemap Processing**: Fetches all API URLs from `https://api.market/sitemap.xml`
- **State Analysis**: Loads previously used APIs from `state.json`
- **API Selection Strategy**:
  - Primary: Selects unused API from sitemap
  - Fallback: If no unused APIs, randomly selects from used APIs
  - Emergency: If no APIs available, exits system
- **Metadata Extraction**: 
  - Attempts to fetch OpenAPI spec (YAML/JSON) from API page
  - Falls back to HTML scraping for title and description
- **Memory Storage**: Stores selected API data in global `api_memory` singleton

**Tracing Details**:
- Records API selection method, source type, pool statistics
- Tracks API metadata (title, description length, OpenAPI availability)
- Logs comprehensive API pool analysis events

### 4. Search Query Generation Phase
**Traced Agent**: `search_agent` with operation `query_generation`

- **AI-Powered Generation**: Uses Gemini 2.5 Pro to generate 5 Reddit-style search queries
- **Query Characteristics**:
  - Natural Reddit language ("How do I...", "Best way to...")
  - Problem-focused rather than API-specific
  - Avoids direct API name references
- **Input**: API title and description from selected API

**Tracing Details**:
- Records query count, API description length
- Stores generated queries as attributes
- Tracks agent capabilities and operation context

### 5. Reddit Search Phase
**Traced Operation**: Tavily search with LangChain integration

- **Search Execution**: Runs each generated query through Tavily search API
- **Query Modification**: Appends `site:reddit.com` to each query
- **URL Filtering**: Extracts only Reddit URLs from search results
- **Deduplication**: Removes duplicate URLs across all queries

**Tracing Details**:
- Uses `NoveumTraceCallbackHandler` for LangChain integration
- Automatically traces Tavily tool execution
- Records search results and URL counts

### 6. Post Validation Phase
**Traced Operation**: `post_validation`

- **Comprehensive Validation**: Checks each Reddit URL for validity
- **Validation Criteria**:
  - API accessibility (200 status code)
  - Post status (not archived, locked, NSFW, quarantined)
  - Author status (not deleted or AutoModerator)
  - Content status (not removed, stickied)
  - Subreddit uniqueness (not previously used)
- **Statistical Tracking**: Maintains detailed rejection statistics
- **Limit Enforcement**: Stops at 5 valid posts per run

**Tracing Details**:
- Records validation statistics (success rate, rejection reasons)
- Logs individual post validation events
- Tracks subreddit usage and post metadata

### 7. Comment Generation Phase
**Traced Agent**: `content_generation_agent` with operation `comment_generation`

- **AI Comment Creation**: Uses Gemini 2.5 Pro to generate authentic Reddit comments
- **Comment Characteristics**:
  - 2-3 sentences, conversational tone
  - Addresses specific post content
  - Naturally includes API URL as helpful resource
  - Avoids promotional language
- **Error Handling**: Continues with other posts if comment generation fails
- **Retry Logic**: Implements exponential backoff for API failures

**Tracing Details**:
- Records successful/failed comment counts
- Stores full comment content in span events
- Tracks generation success rates and error details

### 8. Output Generation Phase
**Traced Operation**: `email_generation_and_sending`

- **HTML Report Creation**: Generates styled HTML report with:
  - API metadata (title, description, URL)
  - Post-comment pairs with subreddit information
  - Copyable comment boxes for easy use
- **Email Delivery**: Sends HTML report via SMTP to configured recipients
- **File Management**: Saves reports to `static/reddit_output_run_{run_number}.html`

**Tracing Details**:
- Records output file paths and sizes
- Tracks email generation and sending events
- Monitors recipient information and service details

### 9. State Management
- **API Tracking**: Marks selected API as used in `state.json`
- **Subreddit Tracking**: Records used subreddits to avoid repetition
- **Persistence**: Maintains state across runs for continuity

### 10. Run Completion
- **Success Criteria**: Valid posts found and comments generated
- **Failure Handling**: Retries with different API if no valid posts found
- **Timing**: Records run duration and completion metrics
- **Trace Closure**: Properly closes trace spans and flushes data

## Tracing Implementation

### Trace Hierarchy
```
reddit_agent_run_{N}
├── api_selection
├── search_agent (query_generation)
│   └── Tavily search (via LangChain callback)
├── post_validation
├── content_generation_agent (comment_generation)
└── email_generation_and_sending
```

### Trace Attributes
Each trace span includes relevant attributes:
- **System-level**: Target runs, max attempts, system type
- **Run-level**: Run number, timing, API details
- **Operation-level**: Counts, success rates, error details
- **Content-level**: Generated queries, comments, validation stats

### Trace Events
Key events are logged throughout execution:
- `api_pool_analyzed`: Complete API availability analysis
- `api_selected`: API selection with method and metadata
- `post_rejected`: Individual post rejection with reason
- `valid_post_found`: Successful post validation
- `comment_generated`: Successful comment creation
- `comment_generation_failed`: Comment generation errors
- `email_generated`: HTML report creation
- `email_sent`: Email delivery confirmation
- `run_completed`: Successful run completion
- `run_failed`: Run failure with reason

### Error Handling and Tracing
- All major operations wrapped in try-catch blocks
- Exceptions logged as trace events with error details
- Graceful degradation (continues with other posts if one fails)
- Comprehensive error context for debugging

### Performance Monitoring
- Start/end timestamps for all operations
- Duration calculations for runs and operations
- Success/failure rates for each phase
- Resource usage tracking (API calls, file sizes)

## Key Features

1. **Intelligent API Selection**: Prioritizes unused APIs with fallback strategies
2. **Natural Query Generation**: AI-powered search queries that sound like real Reddit users
3. **Comprehensive Validation**: Multi-criteria post filtering for quality
4. **Authentic Content**: AI-generated comments that avoid promotional language
5. **State Persistence**: Avoids repetition across runs
6. **Robust Error Handling**: Continues operation despite individual failures
7. **Complete Observability**: Full tracing of every operation and decision

## Output

The system generates:
- HTML reports with styled post-comment pairs
- Email notifications to configured recipients
- Persistent state tracking for continuity
- Comprehensive trace data for monitoring and debugging

This architecture ensures reliable, observable, and maintainable automated Reddit engagement while maintaining authentic, helpful content quality.
