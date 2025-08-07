#!/usr/bin/env python3
"""
Gemini 2.5 Pro Outlined Story Generator
Multi-step story generation: outline creation followed by iterative section writing.
"""

import os
import json
from openai import OpenAI

def read_story_prompt(file_path="story_prompt.txt"):
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

def generate_story_outline(client, story_prompt_content):
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
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "user",
                    "content": outline_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        outline_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if outline_text.startswith('```json'):
            outline_text = outline_text[7:]  # Remove ```json
        elif outline_text.startswith('```'):
            outline_text = outline_text[3:]   # Remove ```
        
        if outline_text.endswith('```'):
            outline_text = outline_text[:-3]  # Remove closing ```
        
        outline_text = outline_text.strip()
        
        # Parse JSON response
        outline_data = json.loads(outline_text)
        
        # Save outline to file
        with open("story_outline.json", "w", encoding="utf-8") as f:
            json.dump(outline_data, f, ensure_ascii=False, indent=2)
        
        print("Story outline generated and saved to story_outline.json")
        return outline_data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {outline_text}")
        return None
    except Exception as e:
        print(f"Error generating outline: {e}")
        return None

def generate_story_section(client, section_info, previous_sections, story_prompt_content):
    """Generate a single story section with context from previous sections."""
    
    # Build context from previous sections
    context = ""
    if previous_sections:
        context = "\n\nPrevious sections of the story:\n"
        for i, prev_section in enumerate(previous_sections, 1):
            context += f"\n--- Section {i} ---\n{prev_section}\n"
    
    section_prompt = f"""You are a skilled Japanese storywriter. Write Section {section_info['section']} of the story in Japanese.

Original Story Prompt:
{story_prompt_content}

Section to write:
Title: {section_info['title']}
Summary: {section_info['summary']}
{context}

Write this section as a complete chapter in Japanese. Make sure it flows naturally from the previous sections and advances the story according to the outline. Write in an engaging, literary style."""

    print(f"Generating Section {section_info['section']}: {section_info['title']}")
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "user",
                    "content": section_prompt
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        section_content = response.choices[0].message.content.strip()
        return section_content
        
    except Exception as e:
        print(f"Error generating section {section_info['section']}: {e}")
        return None

def generate_complete_outlined_story():
    """Generate a complete story using the outline-based approach."""
    
    # Initialize OpenAI client with Gemini endpoint
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Read the story prompt
    story_prompt_content = read_story_prompt()
    
    if not story_prompt_content:
        print("No story prompt found. Please add content to story_prompt.txt")
        return
    
    # Step 1: Generate outline
    outline_data = generate_story_outline(client, story_prompt_content)
    
    if not outline_data:
        print("Failed to generate story outline.")
        return
    
    print(f"\nOutline generated for: {outline_data.get('title', 'Untitled Story')}")
    print("Sections:")
    for section in outline_data['outline']:
        print(f"  {section['section']}. {section['title']} - {section['summary']}")
    
    # Step 2: Generate each section iteratively
    print("\n" + "="*50)
    print("Generating story sections...")
    print("="*50)
    
    complete_story = []
    previous_sections = []
    
    for section_info in outline_data['outline']:
        section_content = generate_story_section(
            client, section_info, previous_sections, story_prompt_content
        )
        
        if section_content:
            complete_story.append(f"## {section_info['title']}\n\n{section_content}")
            previous_sections.append(section_content)
            print(f"✓ Section {section_info['section']} completed")
        else:
            print(f"✗ Failed to generate Section {section_info['section']}")
            break
    
    # Step 3: Save the complete story
    if complete_story:
        story_title = outline_data.get('title', 'Generated Story')
        full_story = f"# {story_title}\n\n" + "\n\n".join(complete_story)
        
        with open("generated_outlined_story.txt", "w", encoding="utf-8") as f:
            f.write(full_story)
        
        print(f"\n✓ Complete story saved to generated_outlined_story.txt")
        print(f"Story contains {len(complete_story)} sections")
        return full_story
    else:
        print("Failed to generate complete story.")
        return None

def main():
    """Main function to run the outlined story generator."""
    print("Japanese Outlined Story Generator using Gemini 2.5 Pro")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not found!")
        return
    
    # Generate the complete story
    story = generate_complete_outlined_story()
    
    if story:
        print("\n" + "="*60)
        print("Story generation completed successfully!")
        print("Files created:")
        print("  - story_outline.json (story outline)")
        print("  - generated_outlined_story.txt (complete story)")
    else:
        print("Story generation failed.")

if __name__ == "__main__":
    main()
