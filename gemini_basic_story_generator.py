#!/usr/bin/env python3
"""
Gemini 2.5 Pro Story Generator
Calls the Gemini API using OpenAI library to generate Japanese stories.
"""

import os
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

def generate_japanese_story():
    """Generate a Japanese story using Gemini 2.5 Pro."""
    
    # Initialize OpenAI client with Gemini endpoint
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Read the story prompt
    story_prompt_content = read_story_prompt()
    
    # Construct the full prompt
    full_prompt = ("You are a skilled Japanese storywriter. Write a chapter of a story in Japanese "
                  "according to the below summary and characters.\n\n" + story_prompt_content)
    
    print("Sending prompt to Gemini 2.5 Pro...")
    print(f"Prompt: {full_prompt}")
    print("-" * 50)
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",  # Use the latest Gemini model available
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        # Extract and return the generated story
        story = response.choices[0].message.content
        return story
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def main():
    """Main function to run the story generator."""
    print("Japanese Story Generator using Gemini 2.5 Pro")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not found!")
        return
    
    # Generate the story
    story = generate_japanese_story()
    
    if story:
        print("\nGenerated Story:")
        print("=" * 50)
        print(story)
        
        # Optionally save to file
        with open("generated_story.txt", "w", encoding="utf-8") as f:
            f.write(story)
        print(f"\nStory saved to generated_story.txt")
    else:
        print("Failed to generate story.")

if __name__ == "__main__":
    main()
