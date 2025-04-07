import json

def json_to_markdown(json_file, output_file):
    """
    Reads a JSON file containing quizzes and converts it to a Markdown file with formatted output.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    md_lines = []
    # Process each quiz
    for quiz_name, questions in data.items():
        md_lines.append(f"# {quiz_name}")
        md_lines.append("")  # Blank line for spacing
        
        # Process each question within the quiz
        for q_num, q_content in questions.items():
            # Use 'question' key if present, otherwise fallback to 'questions'
            question_text = q_content.get("question") or q_content.get("questions", "No question provided")
            
            md_lines.append(f"## Question {q_num}")
            md_lines.append("")
            md_lines.append(f"**Question:** {question_text}")
            md_lines.append("")
            
            # Format options as a bullet list
            options = q_content.get("options", {})
            if options:
                md_lines.append("**Options:**")
                for option, option_text in options.items():
                    md_lines.append(f"- **{option}:** {option_text}")
                md_lines.append("")
            
            # Add the answer
            answer = q_content.get("answer", "No answer provided")
            md_lines.append(f"**Answer:** {answer}")
            md_lines.append("")
            
            # Optionally include professor explanation if available
            professor_explanation = q_content.get("professor_explanation", "").strip()
            if professor_explanation:
                md_lines.append(f"**Professor Explanation:** {professor_explanation}")
                md_lines.append("")
            
            # Include any additional explanation if available
            explanation = q_content.get("explanation", "").strip()
            if explanation:
                md_lines.append(f"**Explanation:** {explanation}")
                md_lines.append("")
            
            # Separator between questions
            md_lines.append("---")
            md_lines.append("")
    
    # Write out the markdown file
    with open(output_file, 'w') as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    # Specify the input JSON file and the output Markdown file.
    json_file = "questions.json"
    output_file = "quiz.md"
    json_to_markdown(json_file, output_file)
