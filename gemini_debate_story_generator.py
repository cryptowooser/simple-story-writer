#!/usr/bin/env python3
"""
Gemini Multi-Agent Debate Story Generator
Uses a three-agent debate system (affirmative, negative, judge) for each story section.
"""

import os
import json
from openai import OpenAI
from typing import Dict, Any, List

class StoryDebateGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.debate_log = []
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "by_agent": {
                "outline_generator": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
                "affirmative_writer": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
                "negative_critic": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
                "judge_editor": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            }
        }
    
    def read_story_prompt(self, file_path="story_prompt.txt"):
        """Read the story prompt from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Using empty prompt.")
            return ""
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def _call_model(self, prompt: str, agent_type: str) -> str:
        """Make API call to Gemini model with token tracking."""
        try:
            print(f"[{agent_type}] Making API call...")
            response = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                reasoning_effort="none"
            )
            
            if not response or not response.choices:
                print(f"[{agent_type}] Error: Empty response from API")
                return None
                
            if not response.choices[0].message:
                print(f"[{agent_type}] Error: No message in response")
                return None
                
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Update totals
                self.token_usage["total_prompt_tokens"] += prompt_tokens
                self.token_usage["total_completion_tokens"] += completion_tokens
                self.token_usage["total_tokens"] += total_tokens
                self.token_usage["api_calls"] += 1
                
                # Update by agent
                if agent_type in self.token_usage["by_agent"]:
                    self.token_usage["by_agent"][agent_type]["prompt_tokens"] += prompt_tokens
                    self.token_usage["by_agent"][agent_type]["completion_tokens"] += completion_tokens
                    self.token_usage["by_agent"][agent_type]["calls"] += 1
                
                print(f"[{agent_type}] Tokens used: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"Error calling model for {agent_type}: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"Response status: {getattr(e.response, 'status_code', 'unknown')}")
            return None
    
    def _clean_json_response(self, response_text: str) -> str:
        """Remove markdown code blocks from JSON responses."""
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def generate_story_outline(self, story_prompt_content: str) -> Dict[str, Any]:
        """Generate a 6-point story outline in JSON format."""
        
        outline_prompt = f"""You are a skilled Japanese storywriter. Based on the story prompt below, create a detailed 6-point outline of the story events in JSON format.

Story Prompt:
{story_prompt_content}

Please provide your response in the following JSON format:
{{
    "title": "Story Title in Japanese",
    "outline": [
        {{
            "section": 1,
            "title": "Section title in Japanese",
            "summary": "Brief summary of what happens in this section"
        }},
        {{
            "section": 2,
            "title": "Section title in Japanese", 
            "summary": "Brief summary of what happens in this section"
        }},
        {{
            "section": 3,
            "title": "Section title in Japanese",
            "summary": "Brief summary of what happens in this section"
        }},
        {{
            "section": 4,
            "title": "Section title in Japanese",
            "summary": "Brief summary of what happens in this section"
        }},
        {{
            "section": 5,
            "title": "Section title in Japanese",
            "summary": "Brief summary of what happens in this section"
        }},
        {{
            "section": 6,
            "title": "Section title in Japanese",
            "summary": "Brief summary of what happens in this section"
        }}
    ]
}}

Respond ONLY with valid JSON. Do not include any other text."""

        print("Generating story outline...")
        
        try:
            response = self._call_model(outline_prompt, "outline_generator")
            if not response:
                return None
            
            outline_text = self._clean_json_response(response)
            outline_data = json.loads(outline_text)
            
            # Save outline to file
            with open("story_outline.json", "w", encoding="utf-8") as f:
                json.dump(outline_data, f, ensure_ascii=False, indent=2)
            
            print("Story outline generated and saved to story_outline.json")
            return outline_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response}")
            return None
        except Exception as e:
            print(f"Error generating outline: {e}")
            return None
    
    def affirmative_writer(self, section_info: Dict, previous_sections: List[str], story_prompt_content: str) -> str:
        """Affirmative agent writes the initial draft of a story section."""
        
        # Build context from previous sections
        context = ""
        if previous_sections:
            context = "\n\nPrevious sections of the story:\n"
            for i, prev_section in enumerate(previous_sections, 1):
                context += f"\n--- Section {i} ---\n{prev_section}\n"
        
        affirmative_prompt = f"""You are a skilled Japanese storywriter acting as the AFFIRMATIVE WRITER. Your role is to write the initial draft of a story section.

Original Story Prompt:
{story_prompt_content}

Section to write:
Title: {section_info['title']}
Summary: {section_info['summary']}
{context}

Write Section {section_info['section']} as a complete chapter in Japanese. Focus on:
- Engaging narrative flow
- Rich character development
- Beautiful prose and literary style
- Consistency with previous sections
- Advancing the story according to the outline

Write in an immersive, literary Japanese style that draws readers in."""

        print(f"Affirmative Writer: Drafting Section {section_info['section']}")
        return self._call_model(affirmative_prompt, "affirmative_writer")
    
    def negative_critic(self, section_info: Dict, affirmative_draft: str, previous_sections: List[str], story_prompt_content: str) -> str:
        """Negative agent critiques and provides alternative version."""
        
        # Build context from previous sections
        context = ""
        if previous_sections:
            context = "\n\nPrevious sections of the story:\n"
            for i, prev_section in enumerate(previous_sections, 1):
                context += f"\n--- Section {i} ---\n{prev_section}\n"
        
        negative_prompt = f"""You are a skilled Japanese storywriter acting as the NEGATIVE CRITIC. You have read the affirmative writer's draft and believe it can be significantly improved.

Original Story Prompt:
{story_prompt_content}

Section Requirements:
Title: {section_info['title']}
Summary: {section_info['summary']}
{context}

AFFIRMATIVE WRITER'S DRAFT:
{affirmative_draft}

As the negative critic, you believe this draft has issues that need addressing. Provide your critique and write an improved alternative version that addresses these problems. Focus on:
- Better narrative pacing and tension
- Deeper character emotions and motivations
- More vivid and evocative descriptions
- Stronger dialogue and character interactions
- Better integration with the overall story arc

Write your improved version of Section {section_info['section']} in Japanese."""

        print(f"Negative Critic: Reviewing and rewriting Section {section_info['section']}")
        return self._call_model(negative_prompt, "negative_critic")
    
    def judge_editor(self, section_info: Dict, affirmative_draft: str, negative_draft: str, previous_sections: List[str]) -> Dict[str, Any]:
        """Judge agent evaluates both versions and selects/refines the final version."""
        
        judge_prompt = f"""You are acting as both a LITERARY EDITOR and JUDGE. Two writers have provided different versions of Section {section_info['section']} titled "{section_info['title']}". Your role is to evaluate both versions and select the superior one, or create a refined version based on the best elements of both.

Section Requirements:
Title: {section_info['title']}
Summary: {section_info['summary']}

AFFIRMATIVE VERSION:
{affirmative_draft}

NEGATIVE VERSION:
{negative_draft}

Evaluate both versions based on:
1. Narrative flow and pacing
2. Character development and emotional depth
3. Prose quality and literary style
4. Consistency with story requirements
5. Reader engagement and immersion

Select the superior version OR create a refined version that combines the best elements. Return your decision in this JSON format:
{{
    "preferred_version": "affirmative" or "negative" or "refined",
    "reasoning": "Detailed explanation of your choice and what makes it superior",
    "final_section": "The complete final version of the section in Japanese"
}}

Respond ONLY with valid JSON."""

        print(f"Judge/Editor: Evaluating Section {section_info['section']}")
        
        try:
            response = self._call_model(judge_prompt, "judge_editor")
            if not response:
                return None
            
            judge_text = self._clean_json_response(response)
            judge_decision = json.loads(judge_text)
            
            # Log the debate for this section
            debate_record = {
                "section": section_info['section'],
                "title": section_info['title'],
                "affirmative_draft": affirmative_draft,
                "negative_draft": negative_draft,
                "judge_decision": judge_decision
            }
            self.debate_log.append(debate_record)
            
            return judge_decision
            
        except json.JSONDecodeError as e:
            print(f"Error parsing judge response: {e}")
            print(f"Raw response: {response}")
            return None
        except Exception as e:
            print(f"Error in judge evaluation: {e}")
            return None
    
    def generate_section_with_debate(self, section_info: Dict, previous_sections: List[str], story_prompt_content: str) -> str:
        """Generate a single section using the three-agent debate process."""
        
        print(f"\n{'='*60}")
        print(f"DEBATE FOR SECTION {section_info['section']}: {section_info['title']}")
        print(f"{'='*60}")
        
        # Step 1: Affirmative writer creates initial draft
        affirmative_draft = self.affirmative_writer(section_info, previous_sections, story_prompt_content)
        if not affirmative_draft:
            print("Failed to generate affirmative draft")
            return None
        
        # Step 2: Negative critic provides alternative
        negative_draft = self.negative_critic(section_info, affirmative_draft, previous_sections, story_prompt_content)
        if not negative_draft:
            print("Failed to generate negative draft")
            return affirmative_draft  # Fallback to affirmative
        
        # Step 3: Judge evaluates and selects final version
        judge_decision = self.judge_editor(section_info, affirmative_draft, negative_draft, previous_sections)
        if not judge_decision:
            print("Failed to get judge decision")
            return affirmative_draft  # Fallback to affirmative
        
        print(f"Judge Decision: {judge_decision['preferred_version']}")
        print(f"Reasoning: {judge_decision['reasoning']}")
        
        return judge_decision['final_section']
    
    def print_token_usage_summary(self):
        """Print a summary of token usage statistics."""
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        print(f"Total API Calls: {self.token_usage['api_calls']}")
        print(f"Total Prompt Tokens: {self.token_usage['total_prompt_tokens']:,}")
        print(f"Total Completion Tokens: {self.token_usage['total_completion_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_tokens']:,}")
        
        print("\nBy Agent:")
        for agent, stats in self.token_usage['by_agent'].items():
            if stats['calls'] > 0:
                total_agent_tokens = stats['prompt_tokens'] + stats['completion_tokens']
                print(f"  {agent.replace('_', ' ').title()}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Tokens: {stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} completion = {total_agent_tokens:,} total")
        
        # Rough cost estimation (these are example rates, adjust based on actual Gemini pricing)
        estimated_cost = (self.token_usage['total_prompt_tokens'] * 0.00001) + (self.token_usage['total_completion_tokens'] * 0.00003)
        print(f"\nEstimated Cost: ${estimated_cost:.4f} USD (approximate)")
        print("="*60)
    
    def save_token_usage_log(self):
        """Save detailed token usage to file."""
        with open("token_usage_log.json", "w", encoding="utf-8") as f:
            json.dump(self.token_usage, f, ensure_ascii=False, indent=2)
        print("Token usage log saved to token_usage_log.json")
    
    def generate_complete_debate_story(self):
        """Generate a complete story using the multi-agent debate approach."""
        
        # Read the story prompt
        story_prompt_content = self.read_story_prompt()
        
        if not story_prompt_content:
            print("No story prompt found. Please add content to story_prompt.txt")
            return
        
        # Step 1: Generate outline
        outline_data = self.generate_story_outline(story_prompt_content)
        
        if not outline_data:
            print("Failed to generate story outline.")
            return
        
        print(f"\nOutline generated for: {outline_data.get('title', 'Untitled Story')}")
        print("Sections:")
        for section in outline_data['outline']:
            print(f"  {section['section']}. {section['title']} - {section['summary']}")
        
        # Step 2: Generate each section using debate process
        print("\n" + "="*80)
        print("STARTING MULTI-AGENT DEBATE STORY GENERATION")
        print("="*80)
        
        complete_story = []
        previous_sections = []
        
        for section_info in outline_data['outline']:
            final_section = self.generate_section_with_debate(
                section_info, previous_sections, story_prompt_content
            )
            
            if final_section:
                complete_story.append(f"## {section_info['title']}\n\n{final_section}")
                previous_sections.append(final_section)
                print(f"✓ Section {section_info['section']} completed through debate")
            else:
                print(f"✗ Failed to generate Section {section_info['section']}")
                break
        
        # Step 3: Save the complete story and debate log
        if complete_story:
            story_title = outline_data.get('title', 'Generated Debate Story')
            full_story = f"# {story_title}\n\n" + "\n\n".join(complete_story)
            
            with open("generated_debate_story.txt", "w", encoding="utf-8") as f:
                f.write(full_story)
            
            # Save debate log
            with open("debate_log.json", "w", encoding="utf-8") as f:
                json.dump(self.debate_log, f, ensure_ascii=False, indent=2)
            
            # Save token usage log
            self.save_token_usage_log()
            
            print(f"\n✓ Complete debate story saved to generated_debate_story.txt")
            print(f"✓ Debate log saved to debate_log.json")
            print(f"✓ Token usage log saved to token_usage_log.json")
            print(f"Story contains {len(complete_story)} sections")
            
            # Print token usage summary
            self.print_token_usage_summary()
            
            return full_story
        else:
            print("Failed to generate complete story.")
            return None

def main():
    """Main function to run the debate story generator."""
    print("Japanese Multi-Agent Debate Story Generator using Gemini 2.5 Pro")
    print("=" * 80)
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not found!")
        return
    
    # Create generator and run
    generator = StoryDebateGenerator()
    story = generator.generate_complete_debate_story()
    
    if story:
        print("\n" + "="*80)
        print("MULTI-AGENT DEBATE STORY GENERATION COMPLETED!")
        print("Files created:")
        print("  - story_outline.json (story outline)")
        print("  - generated_debate_story.txt (final story)")
        print("  - debate_log.json (complete debate records)")
        print("="*80)
    else:
        print("Story generation failed.")

if __name__ == "__main__":
    main()
