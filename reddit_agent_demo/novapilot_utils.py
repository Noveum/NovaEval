"""
NovaPilot Utilities - Agent Analysis and Evaluation Tools

This module provides utilities for analyzing agent performance using Gemini AI
and generating comprehensive reports from evaluation data.
"""

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Prompt templates
SCORER_ANALYSIS_PROMPT = """ A scorer named {scorer_name} is run on different runs of the same agent/llm/tool,
the scorer gives a score and a reasoning, you will be given 25 such samples. Now the scorer may be giving different
reasonings and all of them are in natural language. I want you to highlight the key reasons for your response.
These key reasons will further be used for analysis, and improvement of the agent. Do not suggest any fixes.
Just focus on not missing out on any of the information regarding why the agent is failing.
You have to focus on the low scores only, as we have to improve them.
Some rows might show - "Missing required fields" it is a code issue, on the developer's side, so do not include it in the reasoning.

Just to clarify, your job is not to analyze the scorers, but to analyze the agent. You are basically the representative of the scorers.

Give the reasoning, and start with the scorer name.

In the format - 
Scorer Name: Task Progression
Reasoning: 

"""

AGENTWISE_SUMMARY_PROMPT = """
Different scorers are run on different runs of the same agent/llm/tool, you will be given the reasoning
for each scorer, as to why it gave poor scores. You job is to summarize the information from different
scorers into a single analysis. All the scores are of one specific part of the entire agentic workflow,
so please remove the redundancies that you get. Do not try to suggest fixes, only focus on removing
the redundant information, and keeping the important information.

Just to clarify, your job is not to analyze the summaries, but to analyze the agent. You are basically the representative of the scorers.

In the format -
Agent Name: query_generation
Reasoning: 

"""

FINAL_ANALYSIS_PROMPT = """
An agent is run, and then different scorers are run on specific parts of the agentic workflow.
So if an agent has 5 different parts (llm/tool/agent), and there are 3 scorers, then there will be a total of 15 scores, and respective reasonings. I have condensed these reasonings, in a part wise manner.
You will be given the part wise analysis, and you will also be given the entire agentic workflow, explaining how the agent is set up.

You have to figure out why the agent is failing, you are given a bird's eye view, as in agents, a failure at step 1, may surface at step 3 in the analysis, so you will have to be aware of that.

You have to suggest fixes to the developer in bullet points, in the format ->

Suggested Fixes:
 - fix_1: 
 - fix_2: 
 
"""


class NovaPilotAnalyzer:
    """
    A class for analyzing agent performance using Gemini AI and generating comprehensive reports.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-2.5-pro'):
        """
        Initialize the NovaPilot Analyzer.
        
        Args:
            api_key: Gemini API key. If None, will try to load from environment.
            model_name: Name of the Gemini model to use.
        """
        self._setup_gemini(api_key, model_name)
        self.log_file: Optional[Path] = None
        self.reddit_agent_doc: Optional[str] = None
        
    def _setup_gemini(self, api_key: Optional[str], model_name: str) -> None:
        """Setup Gemini API configuration."""
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def setup_logging(self, log_dir: str = 'log') -> Path:
        """
        Setup logging directory and create timestamped log file.
        
        Args:
            log_dir: Directory to store log files.
            
        Returns:
            Path to the created log file.
        """
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_path / f"analysis_log_{timestamp}.txt"
        
        return self.log_file
        
    def load_agent_documentation(self, doc_path: str) -> str:
        """
        Load agent documentation from a markdown file.
        
        Args:
            doc_path: Path to the agent documentation file.
            
        Returns:
            Content of the documentation file.
        """
        with open(doc_path, 'r', encoding='utf-8') as f:
            self.reddit_agent_doc = f.read()
        return self.reddit_agent_doc
        
    def log_response(self, response: str, description: str) -> None:
        """
        Log a response to the log file.
        
        Args:
            response: The response text to log.
            description: Description of what the response contains.
        """
        if not self.log_file:
            raise ValueError("Logging not setup. Call setup_logging() first.")
            
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DESCRIPTION: {description}\n")
            f.write(f"{'='*50}\n")
            f.write(f"{response}\n")
            f.write(f"{'='*50}\n\n")
            
    def process_csv_file(self, csv_path: str, max_rows: int = 25) -> Dict[str, str]:
        """
        Process CSV file and extract scorer data for specified number of rows.
        
        Args:
            csv_path: Path to the CSV file.
            max_rows: Maximum number of rows to process.
            
        Returns:
            Dictionary with scorer data strings.
        """
        df = pd.read_csv(csv_path)
        
        # Get first max_rows rows (or all if less than max_rows)
        rows_to_process = min(max_rows, len(df))
        df_subset = df.head(rows_to_process)
        
        # Get column names
        columns = df.columns.tolist()
        
        # Skip first 4 columns (IDs)
        # The remaining columns are split into score columns and reasoning columns
        remaining_columns = columns[4:]
        n_scorers = len(remaining_columns) // 2  # Each scorer has score + reasoning column
        
        score_columns = remaining_columns[:n_scorers]  # First half are score columns
        reasoning_columns = remaining_columns[n_scorers:]  # Second half are reasoning columns
        
        scorer_data = {}
        
        # For each scorer
        for i, scorer_name in enumerate(score_columns):
            scorer_strings = []
            reasoning_col = reasoning_columns[i]
            
            for _, row in df_subset.iterrows():
                score = row[scorer_name]
                reasoning = row[reasoning_col]
                scorer_string = f"{scorer_name} score = {score} reasoning = {reasoning}"
                scorer_strings.append(scorer_string)
            
            scorer_data[scorer_name] = "\n".join(scorer_strings)
        
        return scorer_data
        
    def analyze_scorer(self, scorer_name: str, scorer_data: str, dataset_name: str) -> str:
        """
        Analyze a single scorer's data using Gemini AI.
        
        Args:
            scorer_name: Name of the scorer.
            scorer_data: Data string for the scorer.
            dataset_name: Name of the dataset being analyzed.
            
        Returns:
            Analysis response from Gemini AI.
        """
        prompt = f"""{SCORER_ANALYSIS_PROMPT}
        
        Scorer Data: 
        Scorer Name: {scorer_name}
        {scorer_data}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            response_text = "Analysis for scorer: " + scorer_name + "\n" + response_text
            
            # Log the response
            self.log_response(response_text, f"{dataset_name} - {scorer_name} Analysis")
            
            return f"Scorer: {scorer_name}\n{response_text}"
            
        except Exception as e:
            error_msg = f"Error processing {scorer_name}: {str(e)}"
            self.log_response(error_msg, f"{dataset_name} - {scorer_name} Error")
            raise e
            
    def create_dataset_summary(self, dataset_name: str, scorer_analyses: List[str]) -> str:
        """
        Create a summary for a dataset based on scorer analyses.
        
        Args:
            dataset_name: Name of the dataset.
            scorer_analyses: List of scorer analysis strings.
            
        Returns:
            Summary response from Gemini AI.
        """
        combined_responses = "\n\n".join(scorer_analyses)
        
        summary_prompt = f"""{AGENTWISE_SUMMARY_PROMPT}
        
        Agent Name: {dataset_name}
        Scorer Analyses:
        {combined_responses}
        """
        
        try:
            summary_response = self.model.generate_content(summary_prompt)
            summary_text = summary_response.text
            
            # Log the summary
            self.log_response(summary_text, f"{dataset_name} - Dataset Summary")
            
            return f"Dataset: {dataset_name}\n{summary_text}"
            
        except Exception as e:
            error_msg = f"Error creating summary for {dataset_name}: {str(e)}"
            self.log_response(error_msg, f"{dataset_name} - Summary Error")
            raise e
            
    def process_dataset_directory(self, dataset_dir: Path, verbose: bool = True) -> Optional[str]:
        """
        Process a single dataset directory.
        
        Args:
            dataset_dir: Path to the dataset directory.
            verbose: Whether to print progress messages.
            
        Returns:
            Dataset summary string if successful, None if no CSV files found.
        """
        if verbose:
            print(f"\nProcessing {dataset_dir.name}...")
        
        # Find CSV file in the directory
        csv_files = list(dataset_dir.glob('*.csv'))
        if not csv_files:
            if verbose:
                print(f"  No CSV files found in {dataset_dir.name}, skipping...")
            return None
        
        csv_file = csv_files[0]  # Take the first CSV file
        if verbose:
            print(f"  Processing CSV: {csv_file.name}")
        
        # Process the CSV file
        scorer_data = self.process_csv_file(csv_file)
        
        # Store responses for this dataset
        dataset_responses = []
        
        # For each scorer, make a Gemini API call
        for scorer_name, scorer_string in scorer_data.items():
            if verbose:
                print(f"    Making Gemini call for scorer: {scorer_name}")
            
            try:
                analysis = self.analyze_scorer(scorer_name, scorer_string, dataset_dir.name)
                dataset_responses.append(analysis)
            except Exception as e:
                if verbose:
                    print(f"    Error processing {scorer_name}: {str(e)}")
                continue
        
        # Make summary call for this dataset
        if verbose:
            print(f"    Making summary call for {dataset_dir.name}")
        
        try:
            summary = self.create_dataset_summary(dataset_dir.name, dataset_responses)
            return summary
        except Exception as e:
            if verbose:
                print(f"    Error creating summary: {str(e)}")
            return None
            
    def process_all_datasets(self, demo_results_dir: str = 'demo_results', verbose: bool = True) -> List[str]:
        """
        Process all dataset directories in the demo results folder.
        
        Args:
            demo_results_dir: Path to the demo results directory.
            verbose: Whether to print progress messages.
            
        Returns:
            List of dataset summary strings.
        """
        demo_results_path = Path(demo_results_dir)
        all_summaries = []
        
        # Get all dataset directories
        dataset_dirs = [d for d in demo_results_path.iterdir() if d.is_dir()]
        
        if verbose:
            print(f"Found {len(dataset_dirs)} dataset directories to process:")
            for d in dataset_dirs:
                print(f"  - {d.name}")
        
        # Process each dataset directory
        for dataset_dir in dataset_dirs:
            summary = self.process_dataset_directory(dataset_dir, verbose)
            if summary:
                all_summaries.append(summary)
        
        if verbose:
            print(f"\nCompleted processing {len(dataset_dirs)} datasets.")
            
        return all_summaries
        
    def create_final_analysis(self, dataset_summaries: List[str], agent_doc: Optional[str] = None) -> str:
        """
        Create final comprehensive analysis from all dataset summaries.
        
        Args:
            dataset_summaries: List of dataset summary strings.
            agent_doc: Agent documentation string. If None, uses loaded documentation.
            
        Returns:
            Final analysis response from Gemini AI.
        """
        if agent_doc is None:
            agent_doc = self.reddit_agent_doc
            
        if not agent_doc:
            raise ValueError("No agent documentation provided. Load it first or pass as parameter.")
        
        # Combine all dataset summaries
        combined_summaries = "\n\n".join(dataset_summaries)
        
        final_prompt = f"""{FINAL_ANALYSIS_PROMPT}

        Reddit Agent Documentation:
        {agent_doc}

        Dataset Summaries:
        {combined_summaries}
        """
        
        try:
            final_response = self.model.generate_content(final_prompt)
            final_text = final_response.text
            
            # Log the final analysis
            self.log_response(final_text, "Final Comprehensive Analysis")
            
            return final_text
            
        except Exception as e:
            error_msg = f"Error creating final analysis: {str(e)}"
            self.log_response(error_msg, "Final Analysis Error")
            raise e
            
    def run_complete_analysis(self, 
                            demo_results_dir: str = 'demo_results',
                            agent_doc_path: Optional[str] = None,
                            log_dir: str = 'log',
                            verbose: bool = True) -> Tuple[str, List[str], Path]:
        """
        Run the complete analysis pipeline.
        
        Args:
            demo_results_dir: Path to the demo results directory.
            agent_doc_path: Path to agent documentation file.
            log_dir: Directory to store log files.
            verbose: Whether to print progress messages.
            
        Returns:
            Tuple of (final_analysis, dataset_summaries, log_file_path).
        """
        # Setup logging
        log_file = self.setup_logging(log_dir)
        
        # Load agent documentation if provided
        if agent_doc_path:
            self.load_agent_documentation(agent_doc_path)
        
        # Process all datasets
        dataset_summaries = self.process_all_datasets(demo_results_dir, verbose)
        
        # Create final analysis
        if verbose:
            print("\nMaking final comprehensive analysis call...")
            
        final_analysis = self.create_final_analysis(dataset_summaries)
        
        if verbose:
            print("Final analysis completed and logged!")
            print(f"All responses have been logged to: {log_file}")
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE!")
            print("="*50)
            print(f"Log file location: {log_file}")
            print(f"Total datasets processed: {len(dataset_summaries)}")
            print(f"Total summaries generated: {len(dataset_summaries)}")
            print("="*50)
        
        return final_analysis, dataset_summaries, log_file


# Convenience functions for backward compatibility
def setup_gemini_analyzer(api_key: Optional[str] = None, model_name: str = 'gemini-2.5-pro') -> NovaPilotAnalyzer:
    """
    Create and setup a NovaPilot Analyzer instance.
    
    Args:
        api_key: Gemini API key. If None, will try to load from environment.
        model_name: Name of the Gemini model to use.
        
    Returns:
        Configured NovaPilotAnalyzer instance.
    """
    return NovaPilotAnalyzer(api_key, model_name)


def run_agent_analysis(demo_results_dir: str = 'demo_results',
                      agent_doc_path: Optional[str] = None,
                      log_dir: str = 'log',
                      api_key: Optional[str] = None,
                      model_name: str = 'gemini-2.5-pro',
                      verbose: bool = True) -> Tuple[str, List[str], Path]:
    """
    Run complete agent analysis pipeline.
    
    Args:
        demo_results_dir: Path to the demo results directory.
        agent_doc_path: Path to agent documentation file.
        log_dir: Directory to store log files.
        api_key: Gemini API key. If None, will try to load from environment.
        model_name: Name of the Gemini model to use.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (final_analysis, dataset_summaries, log_file_path).
    """
    analyzer = setup_gemini_analyzer(api_key, model_name)
    return analyzer.run_complete_analysis(demo_results_dir, agent_doc_path, log_dir, verbose)


def recommend_improvements(demo_results_dir: str = 'demo_results',
                          agent_doc_path: str = 'reddit_agent.md',
                          log_dir: str = 'log',
                          api_key: Optional[str] = None,
                          model_name: str = 'gemini-2.5-pro',
                          verbose: bool = True) -> Tuple[str, List[str], Path]:
    """
    Single function to run the complete agent analysis pipeline.
    This is equivalent to running the entire complete_analysis_demo.ipynb notebook.
    
    This function:
    1. Sets up Gemini API and logging
    2. Loads agent documentation
    3. Processes all dataset directories
    4. Analyzes each scorer's data
    5. Creates dataset summaries
    6. Generates final comprehensive analysis with improvement recommendations
    
    Args:
        demo_results_dir: Path to the demo results directory containing dataset folders.
        agent_doc_path: Path to the agent documentation markdown file.
        log_dir: Directory to store analysis log files.
        api_key: Gemini API key. If None, will try to load from environment.
        model_name: Name of the Gemini model to use.
        verbose: Whether to print progress messages.
        
    Returns:
        Tuple of (final_analysis, dataset_summaries, log_file_path).
        - final_analysis: Complete analysis with suggested fixes
        - dataset_summaries: List of summaries for each dataset
        - log_file_path: Path to the generated log file
        
    Example:
        >>> final_analysis, summaries, log_file = recommend_improvements("demo_results")
        >>> print(final_analysis)  # Shows suggested fixes for the agent
    """
    if verbose:
        print("="*60)
        print("NOVAPILOT AGENT ANALYSIS - RECOMMEND IMPROVEMENTS")
        print("="*60)
        print("This function runs the complete analysis pipeline equivalent to")
        print("running the entire complete_analysis_demo.ipynb notebook.")
        print("="*60)
    
    # Initialize analyzer
    analyzer = NovaPilotAnalyzer(api_key, model_name)
    
    # Setup logging
    log_file = analyzer.setup_logging(log_dir)
    if verbose:
        print(f"Setup complete! Log file: {log_file}")
    
    # Load agent documentation
    if os.path.exists(agent_doc_path):
        agent_doc = analyzer.load_agent_documentation(agent_doc_path)
        if verbose:
            print(f"Agent document loaded: {len(agent_doc)} characters")
    else:
        if verbose:
            print(f"Warning: Agent document not found at {agent_doc_path}")
        agent_doc = None
    
    # Process all datasets
    dataset_summaries = analyzer.process_all_datasets(demo_results_dir, verbose)
    
    # Create final comprehensive analysis
    if verbose:
        print("\nMaking final comprehensive analysis call...")
    
    final_analysis = analyzer.create_final_analysis(dataset_summaries, agent_doc)
    
    if verbose:
        print("Final analysis completed and logged!")
        print(f"All responses have been logged to: {log_file}")
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"Log file location: {log_file}")
        print(f"Total datasets processed: {len(dataset_summaries)}")
        print(f"Total summaries generated: {len(dataset_summaries)}")
        print("="*50)
    
    return final_analysis, dataset_summaries, log_file
