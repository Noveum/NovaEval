# Agent Scorers: Past Conversation History Requirements

This document identifies which agent scorers require access to past conversation history (trace) and what specific fields they need from the trajectory record, based on industry standards and frameworks like DeepEval.

## Executive Summary

Based on comprehensive research from DeepEval, LangChain, and other evaluation frameworks, here are the scorers that require past conversation history:

**Scorers Requiring Past Context (16+ total):**
1. **Conversation Coherence Scorer**
2. **Conversation Relevancy Scorer**
3. **Knowledge Retention Scorer**
4. **Conversation Completeness Scorer**
5. **Role Adherence Scorer**
6. **Sentiment Analysis Scorer**
7. **Engagement Level Scorer**
8. **User Satisfaction Prediction Scorer**
9. **Topic Coverage Scorer**
10. **Response Time Analysis Scorer**
11. **Resolution Rate Scorer**
12. **Contextual Precision Scorer**
13. **Contextual Faithfulness Scorer**
14. **Tool Usage Pattern Scorer**
15. **Tool Selection Consistency Scorer**
16. **RAG Context Continuity Scorer**

**Scorers NOT Requiring Past Context (5 total):**
1. **Tool Call Relevancy Scorer**
2. **Tool Call Correctness Scorer**
3. **Parameter Correctness Scorer**
4. **Task Progression Scorer**
5. **Context Relevancy Scorer** (current-turn only)

**Note:** The initial identification of only 5 context-dependent scorers was coincidental - comprehensive research reveals many more specialized evaluation metrics that require conversation history.

## Detailed Scorer Analysis

### Scorers That **DO NOT** Require Past Context

These scorers work with current-turn information only:

#### 1. Tool Call Relevancy Scorer
- **Purpose**: Evaluates if tools invoked are relevant to the current user query
- **Required Fields**: 
  - `current.tool_calls`
  - `current.user_query`
  - `available_tools`
- **Context Dependency**: ❌ No past context needed

#### 2. Tool Call Correctness Scorer
- **Purpose**: Assesses if tools are used correctly based on current context
- **Required Fields**:
  - `current.tool_calls`
  - `current.expected_tool_calls`
- **Context Dependency**: ❌ No past context needed

#### 3. Parameter Correctness Scorer
- **Purpose**: Checks if parameters passed to tools are accurate
- **Required Fields**:
  - `current.tool_calls`
  - `current.parameters`
  - `tool_results`
- **Context Dependency**: ❌ No past context needed

#### 4. Task Progression Scorer
- **Purpose**: Evaluates if agent is making progress towards completing current task
- **Required Fields**:
  - `current.task`
  - `current.agent_response`
- **Context Dependency**: ❌ No past context needed

#### 5. Context Relevancy Scorer (Current-Turn)
- **Purpose**: Assesses if agent's response is relevant to current context
- **Required Fields**:
  - `current.context`
  - `current.agent_response`
- **Context Dependency**: ❌ No past context needed

### Scorers That **DO** Require Past Context

#### Core Conversational Scorers (DeepEval Standard)

#### 1. Conversation Coherence Scorer ✅
- **Purpose**: Assesses the logical flow and consistency of the conversation
- **Source**: Industry standard, used in multiple frameworks
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].speaker`
  - `conversation.turns[].message`
  - `conversation.context`

#### 2. Conversation Relevancy Scorer ✅
- **Purpose**: Determines if chatbot's responses are relevant throughout the conversation
- **Source**: DeepEval, industry standard
- **Context Dependency**: ✅ **Requires conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].input`
  - `conversation.turns[].actual_output`

#### 3. Knowledge Retention Scorer ✅
- **Purpose**: Measures chatbot's ability to retain and utilize information presented earlier
- **Source**: DeepEval, industry standard
- **Context Dependency**: ✅ **Requires conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].input`
  - `conversation.turns[].actual_output`

#### 4. Conversation Completeness Scorer ✅
- **Purpose**: Determines if chatbot fulfills user requests throughout the conversation
- **Source**: DeepEval
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].input`
  - `conversation.turns[].actual_output`

#### 5. Role Adherence Scorer ✅
- **Purpose**: Evaluates if chatbot maintains its designated role during the conversation
- **Source**: DeepEval
- **Context Dependency**: ✅ **Requires conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].input`
  - `conversation.turns[].actual_output`
  - `conversation.chatbot_role`

#### Advanced Conversational Scorers

#### 6. Sentiment Analysis Scorer ✅
- **Purpose**: Analyzes emotional tone of the conversation over time
- **Source**: Industry standard, customer service frameworks
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].message`
  - `conversation.turns[].sentiment_score`
  - `conversation.turns[].timestamp`

#### 7. Engagement Level Scorer ✅
- **Purpose**: Measures user engagement and interaction patterns throughout conversation
- **Source**: User experience frameworks, chatbot analytics
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].message`
  - `conversation.turns[].response_time`
  - `conversation.turns[].message_length`
  - `conversation.engagement_metrics`

#### 8. User Satisfaction Prediction Scorer ✅
- **Purpose**: Predicts user satisfaction based on entire conversation patterns
- **Source**: Customer service AI, satisfaction analytics
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].input`
  - `conversation.turns[].actual_output`
  - `conversation.turns[].sentiment_score`
  - `conversation.satisfaction_indicators`

#### 9. Topic Coverage Scorer ✅
- **Purpose**: Assesses whether all user topics and questions have been addressed
- **Source**: Educational AI, comprehensive support systems
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].topics`
  - `conversation.turns[].intents`
  - `conversation.initial_topics`
  - `conversation.resolved_topics`

#### 10. Response Time Analysis Scorer ✅
- **Purpose**: Evaluates response timeliness patterns and consistency
- **Source**: Performance monitoring, SLA compliance frameworks
- **Context Dependency**: ✅ **Requires conversation history with timing**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].timestamp`
  - `conversation.turns[].response_time`
  - `conversation.sla_requirements`

#### 11. Resolution Rate Scorer ✅
- **Purpose**: Determines conversation effectiveness in resolving user issues
- **Source**: Customer support analytics, issue tracking systems
- **Context Dependency**: ✅ **Requires full conversation history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.initial_issue`
  - `conversation.resolution_status`
  - `conversation.escalation_events`
  - `conversation.outcome`

#### RAG-Specific Context Scorers

#### 12. Contextual Precision Scorer ✅
- **Purpose**: Measures relevance of retrieved context across conversation
- **Source**: RAG evaluation frameworks, information retrieval
- **Context Dependency**: ✅ **Requires conversation + retrieval history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].retrieved_context`
  - `conversation.turns[].ground_truth`
  - `conversation.turns[].context_relevance_score`

#### 13. Contextual Faithfulness Scorer ✅
- **Purpose**: Assesses how well retrieved context supports answers over time
- **Source**: RAG evaluation frameworks, factual accuracy systems
- **Context Dependency**: ✅ **Requires conversation + retrieval history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].retrieved_context`
  - `conversation.turns[].agent_response`
  - `conversation.turns[].faithfulness_score`

#### Tool & RAG History Scorers

#### 14. Tool Usage Pattern Scorer ✅
- **Purpose**: Evaluates patterns and efficiency of tool usage across conversation
- **Source**: Agent evaluation frameworks, tool usage analytics
- **Context Dependency**: ✅ **Requires full tool call history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].tool_calls[]`
  - `conversation.turns[].tool_calls[].name`
  - `conversation.turns[].tool_calls[].parameters`
  - `conversation.turns[].tool_calls[].result`
  - `conversation.turns[].tool_calls[].execution_time`
  - `conversation.tool_usage_patterns`

#### 15. Tool Selection Consistency Scorer ✅
- **Purpose**: Assesses consistency in tool selection for similar tasks across conversation
- **Source**: Agent behavior analysis, tool optimization frameworks
- **Context Dependency**: ✅ **Requires full tool call history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].tool_calls[]`
  - `conversation.turns[].task_type`
  - `conversation.turns[].tool_selection_rationale`
  - `conversation.available_tools`
  - `conversation.tool_selection_patterns`

#### 16. RAG Context Continuity Scorer ✅
- **Purpose**: Evaluates how well RAG context builds upon previous retrievals across turns
- **Source**: RAG evaluation frameworks, knowledge continuity systems
- **Context Dependency**: ✅ **Requires full RAG retrieval history**
- **Required Fields**:
  - `conversation.turns[]`
  - `conversation.turns[].retrieved_context`
  - `conversation.turns[].retrieval_query`
  - `conversation.turns[].context_sources`
  - `conversation.turns[].context_overlap_score`
  - `conversation.knowledge_continuity_score`

## Union of All Required Trace Fields

Based on industry standards (DeepEval, LangChain, OpenAI) and the analysis above, here are **all the fields that need to be present in the trace** to support context-dependent scorers:

### Standard Trace Structure (Based on DeepEval & Industry Standards)

```json
{
  "conversation": {
    "turns": [
      {
        "role": "user|assistant", 
        "content": "message content",
        "input": "user input (for user turns)",
        "actual_output": "agent response (for assistant turns)",
        "timestamp": "ISO timestamp (optional)",
        "tool_calls": [
          {
            "name": "tool_name",
            "parameters": {...},
            "result": "tool_result"
          }
        ]
      }
    ],
    "chatbot_role": "designated role/persona of the agent",
    "scenario": "conversation context/situation description",
    "expected_outcome": "desired result of the conversation"
  }
}
```

### Required Fields Summary

#### Essential Fields (Required by All Context-Dependent Scorers):
1. **`conversation.turns[]`** - Array of conversation exchanges
2. **`conversation.turns[].role`** - Speaker identification ("user" or "assistant")  
3. **`conversation.turns[].content`** - Message content (universal field)

#### Additional Required Fields (Per Scorer):
4. **`conversation.turns[].input`** - User input (required by Relevancy, Completeness, Knowledge Retention, Role Adherence)
5. **`conversation.turns[].actual_output`** - Agent response (required by Relevancy, Completeness, Knowledge Retention, Role Adherence)
6. **`conversation.chatbot_role`** - Agent's designated role (required by Role Adherence)
7. **`conversation.turns[].speaker`** - Alternative to "role" (required by Coherence)
8. **`conversation.turns[].message`** - Alternative to "content" (required by Coherence)
9. **`conversation.context`** - Role/system context (required by Coherence)

#### Optional but Recommended Fields:
10. **`conversation.turns[].timestamp`** - Temporal ordering
11. **`conversation.scenario`** - Context description for better evaluation
12. **`conversation.expected_outcome`** - Conversation goals
13. **`conversation.turns[].tool_calls[]`** - Tool usage tracking for agent actions

### Unified Field Mapping

To support all scorers, your trace should include these fields with this mapping:

```json
{
  "conversation": {
    "turns": [
      {
        // Universal fields (support all scorers)
        "role": "user|assistant",           // Maps to both "role" and "speaker" 
        "content": "message text",          // Maps to both "content" and "message"
        
        // Specific fields for certain scorers  
        "input": "user input text",         // For Relevancy, Completeness, Knowledge Retention, Role Adherence
        "actual_output": "agent response",  // For Relevancy, Completeness, Knowledge Retention, Role Adherence
        
        // Optional fields
        "timestamp": "2024-01-01T10:00:00Z",
        "tool_calls": [...]
      }
    ],
    
    // Conversation-level metadata
    "chatbot_role": "agent role description",    // Required by Role Adherence
    "context": "system/role context",           // Required by Coherence
    "scenario": "conversation situation",       // Optional but recommended
    "expected_outcome": "desired result"        // Optional but recommended
  }
}
```

### Data Types and Constraints

- **`conversation.turns`**: `Array` - Chronologically ordered conversation exchanges
- **`conversation.turns[].role`**: `string` - Must be "user" or "assistant"
- **`conversation.turns[].content`**: `string` - Non-empty message content
- **`conversation.turns[].input`**: `string` - User input for that turn
- **`conversation.turns[].actual_output`**: `string` - Agent response for that turn
- **`conversation.chatbot_role`**: `string` - Clear role description
- **`conversation.turns[].timestamp`**: `string` - ISO 8601 format recommended
- **`conversation.turns[].tool_calls`**: `Array` - Tool usage information

## Implementation Notes

1. **Field Redundancy**: Some scorers use different field names for the same data (e.g., `role` vs `speaker`, `content` vs `message`). Include both to ensure compatibility.

2. **Turn-Level vs Message-Level**: Some scorers expect `input`/`actual_output` per turn, others expect `content` per message. The unified structure supports both approaches.

3. **Performance Considerations**: Full conversation history analysis can be computationally expensive. Consider implementing caching and incremental analysis for large traces.

4. **Data Validation**: Ensure proper validation of conversation turn structure and speaker identification to prevent scoring errors.

5. **Tool Call Integration**: While not required by conversation scorers, including `tool_calls` in turns enables comprehensive agent behavior analysis.

## Key Takeaways

### Scorers Requiring Past Context (5 total):
- **Conversation Coherence**: Needs `turns[].speaker`, `turns[].message`, `context`
- **Conversation Relevancy**: Needs `turns[].input`, `turns[].actual_output`  
- **Knowledge Retention**: Needs `turns[].input`, `turns[].actual_output`
- **Conversation Completeness**: Needs `turns[].input`, `turns[].actual_output`
- **Role Adherence**: Needs `turns[].input`, `turns[].actual_output`, `chatbot_role`

### Scorers NOT Requiring Past Context (5 total):
- **Tool Call Relevancy**: Current tool calls + available tools only
- **Tool Call Correctness**: Current vs expected tool calls only
- **Parameter Correctness**: Current parameters + results only  
- **Task Progression**: Current task + response only
- **Context Relevancy**: Current context + response only

### Essential Trace Fields (Union of All Requirements):

Based on all 13+ context-dependent scorers, here's the comprehensive trace structure:

```json
{
  "conversation": {
    "turns": [
      {
        // Universal fields (required by most scorers)
        "role": "user|assistant",           // Required by all
        "content": "message content",       // Required by all
        "input": "user input",             // Required by 8+ scorers
        "actual_output": "agent response", // Required by 8+ scorers
        "timestamp": "ISO timestamp",      // Required by timing scorers
        
        // Advanced analysis fields
        "sentiment_score": 0.8,            // For sentiment analysis
        "response_time": 1.2,              // For response time analysis
        "message_length": 150,             // For engagement scoring
        "topics": ["topic1", "topic2"],    // For topic coverage
        "intents": ["intent1"],            // For completeness scoring
        
        // Tool call fields (for tool pattern analysis)
        "tool_calls": [                   // For tool usage pattern scoring
          {
            "name": "tool_name",
            "parameters": {...},
            "result": "tool_output",
            "execution_time": 1.5,
            "success": true
          }
        ],
        "task_type": "data_analysis",      // For tool selection consistency
        "tool_selection_rationale": "...", // For tool selection consistency
        
        // RAG-specific fields
        "retrieved_context": "...",        // For contextual precision/faithfulness/continuity
        "retrieval_query": "search query", // For RAG context continuity
        "context_sources": ["source1"],    // For RAG context continuity
        "ground_truth": "...",             // For RAG evaluation
        "context_relevance_score": 0.9,    // For contextual precision
        "faithfulness_score": 0.85,        // For contextual faithfulness
        "context_overlap_score": 0.7,      // For RAG context continuity
        
        // Alternative naming (for compatibility)
        "speaker": "user|assistant",        // Alternative to "role"
        "message": "message content"        // Alternative to "content"
      }
    ],
    
    // Conversation-level metadata
    "chatbot_role": "agent role description",     // For role adherence
    "context": "system/role context",            // Alternative to chatbot_role
    "initial_issue": "user's initial problem",   // For resolution tracking
    "initial_topics": ["topic1", "topic2"],     // For topic coverage
    "resolved_topics": ["topic1"],              // For topic coverage
    
    // Advanced conversation metadata
    "satisfaction_indicators": {...},           // For satisfaction prediction
    "engagement_metrics": {...},               // For engagement scoring
    "resolution_status": "resolved|escalated",  // For resolution rate
    "escalation_events": [...],                 // For resolution tracking
    "outcome": "success|failure|partial",       // For resolution scoring
    "sla_requirements": {...},                  // For response time analysis
    
    // Tool & RAG conversation-level metadata
    "available_tools": [...],                   // For tool selection consistency
    "tool_usage_patterns": {...},              // For tool usage pattern analysis
    "tool_selection_patterns": {...},          // For tool selection consistency
    "knowledge_continuity_score": 0.8,         // For RAG context continuity
    
    // Optional but recommended
    "scenario": "conversation situation",       // For better context
    "expected_outcome": "desired result"        // For goal-oriented evaluation
  }
}
```

### Field Priority Levels:

**Essential (Required by 8+ scorers):**
- `conversation.turns[].role`
- `conversation.turns[].content` 
- `conversation.turns[].input`
- `conversation.turns[].actual_output`
- `conversation.turns[].timestamp`
- `conversation.chatbot_role`

**Important (Required by 3-7 scorers):**
- `conversation.turns[].sentiment_score`
- `conversation.turns[].response_time`
- `conversation.turns[].topics`
- `conversation.turns[].tool_calls[]` (for tool pattern analysis)
- `conversation.initial_issue`
- `conversation.resolution_status`

**Tool & RAG Specific (Required by tool/RAG scorers):**
- `conversation.turns[].tool_calls[].name`
- `conversation.turns[].tool_calls[].parameters`
- `conversation.turns[].tool_calls[].result`
- `conversation.turns[].tool_calls[].execution_time`
- `conversation.turns[].retrieved_context`
- `conversation.turns[].retrieval_query`
- `conversation.turns[].context_sources`
- `conversation.available_tools`
- `conversation.tool_usage_patterns`

**Specialized (Required by 1-2 scorers):**
- `conversation.turns[].faithfulness_score`
- `conversation.turns[].context_overlap_score`
- `conversation.turns[].tool_selection_rationale`
- `conversation.engagement_metrics`
- `conversation.sla_requirements`

This comprehensive structure supports all 16+ context-dependent scorers while maintaining compatibility with existing frameworks like DeepEval, LangChain, and specialized evaluation systems.

## **Key Insight: Tool Calls & RAG History ARE Critical**

You were absolutely right to question this! **Tool calls and RAG outputs from previous turns are essential** for several important scorers:

### **Tool History Scorers:**
- **Tool Usage Pattern Scorer**: Analyzes efficiency and patterns of tool usage across the entire conversation
- **Tool Selection Consistency Scorer**: Ensures similar tasks use consistent tool choices throughout the dialogue

### **RAG History Scorers:**  
- **RAG Context Continuity Scorer**: Evaluates how well retrieved context builds upon previous retrievals
- **Contextual Precision/Faithfulness**: Assess RAG quality across multiple turns

### **Why This Matters:**
- **Tool patterns** reveal agent learning and optimization over time
- **RAG continuity** shows how well knowledge compounds across conversation turns
- **Historical context** enables evaluation of agent improvement within a single session
- **Cross-turn consistency** is crucial for user trust and experience quality

The trace structure now properly includes comprehensive tool call and RAG fields to support these advanced evaluation scenarios!

## **Detailed Analysis: Scorers That Use Tool Calls & RAG History**

The following scorers specifically analyze tool calls and RAG outputs from previous conversation turns. Understanding what they evaluate helps design better agent systems:

### **Tool Call History Scorers**

#### **Tool Usage Pattern Scorer**
**What it analyzes:**
- **Tool call sequences** across the entire conversation
- **Frequency patterns** of tool usage (overuse vs. underuse)
- **Tool switching behavior** (does agent jump between tools unnecessarily?)
- **Execution efficiency** (response times, success rates over time)

**What it's looking for:**
- ✅ **Consistent performance**: Tools execute reliably across turns
- ✅ **Learning patterns**: Agent gets better at tool usage over time
- ✅ **Efficiency trends**: Faster execution or fewer failed attempts as conversation progresses
- ❌ **Tool thrashing**: Excessive switching between tools for similar tasks
- ❌ **Performance degradation**: Tools getting slower or less reliable over time

**Example evaluation:**
```
Turn 1: calculator(2+2) → 4 (1.2s)
Turn 5: calculator(15*3) → 45 (0.8s)  ✅ Improved speed
Turn 8: calculator(100/4) → 25 (0.7s)  ✅ Consistent improvement
```

#### **Tool Selection Consistency Scorer**
**What it analyzes:**
- **Task-to-tool mapping** consistency across conversation
- **Similar task handling** (does agent use same tools for similar requests?)
- **Tool selection rationale** evolution over time
- **Alternative tool consideration** patterns

**What it's looking for:**
- ✅ **Consistent choices**: Similar tasks → same optimal tools
- ✅ **Logical progression**: Tool choices become more refined over time
- ✅ **Context awareness**: Tool selection considers previous successful patterns
- ❌ **Inconsistent mapping**: Same task type uses different tools randomly
- ❌ **Regression**: Previously good tool choices abandoned without reason

**Example evaluation:**
```
Turn 2: Data analysis task → pandas_analyzer ✅
Turn 6: Data analysis task → pandas_analyzer ✅ Consistent
Turn 9: Data analysis task → basic_calculator ❌ Inconsistent regression
```

### **RAG History Scorers**

#### **RAG Context Continuity Scorer**
**What it analyzes:**
- **Context building** across multiple retrievals
- **Information overlap** between consecutive RAG calls
- **Knowledge gap filling** patterns
- **Context relevance evolution** throughout conversation

**What it's looking for:**
- ✅ **Cumulative knowledge**: New retrievals build upon previous context
- ✅ **Gap identification**: RAG fills missing information from earlier turns
- ✅ **Context coherence**: Retrieved information forms coherent knowledge base
- ✅ **Redundancy avoidance**: Doesn't re-retrieve identical information
- ❌ **Context fragmentation**: Retrieved info contradicts previous context
- ❌ **Knowledge loops**: Repeatedly retrieves same irrelevant information

**Example evaluation:**
```
Turn 1: RAG retrieves "Python basics" for coding question
Turn 3: RAG retrieves "Python advanced functions" (builds on basics) ✅
Turn 5: RAG retrieves "Python debugging" (completes knowledge) ✅
Turn 7: RAG retrieves "Java basics" (unrelated to conversation) ❌
```

#### **Contextual Precision Scorer (Cross-Turn)**
**What it analyzes:**
- **Retrieval quality improvement** over conversation turns
- **Query refinement** based on previous retrieval results
- **Context relevance** relative to accumulated conversation knowledge

**What it's looking for:**
- ✅ **Query evolution**: Retrieval queries become more specific over time
- ✅ **Precision improvement**: Later retrievals more targeted and relevant
- ✅ **Context awareness**: Retrievals consider what was already found
- ❌ **Query stagnation**: Same broad queries regardless of previous results
- ❌ **Precision degradation**: Later retrievals become less relevant

#### **Contextual Faithfulness Scorer (Cross-Turn)**
**What it analyzes:**
- **Consistency** between retrieved context and agent responses over time
- **Context utilization** patterns across multiple turns
- **Information synthesis** quality as context accumulates

**What it's looking for:**
- ✅ **Faithful integration**: Agent responses accurately reflect accumulated context
- ✅ **Synthesis quality**: Multiple context pieces combined coherently
- ✅ **Consistency maintenance**: No contradictions between turns using different context
- ❌ **Context drift**: Agent responses diverge from retrieved information over time
- ❌ **Synthesis failure**: Cannot combine multiple context pieces effectively

### **Why These Scorers Matter for Agent Development**

#### **For Tool-Based Agents:**
- **Performance optimization**: Identify which tools work best in which contexts
- **Learning detection**: Verify agents improve tool usage over time
- **Efficiency gains**: Spot opportunities to reduce tool call overhead
- **Consistency assurance**: Ensure reliable user experience across conversations

#### **For RAG-Based Agents:**
- **Knowledge building**: Verify agents build coherent knowledge bases over conversations
- **Retrieval optimization**: Identify when retrieval strategies improve or degrade
- **Context management**: Ensure retrieved information enhances rather than confuses responses
- **Information synthesis**: Validate ability to combine multiple knowledge sources effectively

#### **For System Design:**
- **Tool selection algorithms**: Data to improve automated tool selection
- **RAG strategy optimization**: Insights for better retrieval and context management
- **Agent training**: Feedback for improving agent decision-making patterns
- **User experience**: Ensure consistent, improving performance across conversation length

These scorers provide crucial insights that single-turn evaluations miss, enabling development of agents that truly learn and improve within conversations rather than treating each interaction in isolation.
