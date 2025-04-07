import os
import re
import json

def gather_resources(resources_dir):
    """
    Scans the specified directory for files that match the naming scheme:
    Quiz{x}-{y}-{section_name}.{ext}
    
    Returns a nested dictionary of the form:
    resources[quiz_num][question_num][section_name] = list of (filepath, extension)
    """
    pattern = re.compile(r"^Quiz(\d+)-(\d+)-(.+)\.(png|c|cu)$", re.IGNORECASE)
    resources = {}
    
    for filename in os.listdir(resources_dir):
        match = pattern.match(filename)
        if match:
            quiz_num = match.group(1)
            question_num = match.group(2)
            section_name = match.group(3)       # e.g. "question", "options-A", "professor_explanation", etc.
            file_ext = match.group(4).lower()  # e.g. "png", "c", or "cu"
            
            # Initialize nested dict if needed
            resources.setdefault(quiz_num, {}).setdefault(question_num, {}).setdefault(section_name, [])
            
            # Store the file path (relative to resources_dir) and extension
            resources[quiz_num][question_num][section_name].append((os.path.join(resources_dir, filename), file_ext))
    
    return resources

def embed_resource(filepath, file_ext):
    """
    Returns a string that embeds the resource in Markdown format.
    - If .png, returns an image link.
    - If .c or .cu, reads the file content and wraps it in a Markdown code block.
    """
    if file_ext == "png":
        # Embed as an image
        return f"![Resource image]({filepath})"
    else:
        # Assume .c or .cu => embed code
        # Choose a code fence language. For .c => 'c', for .cu => 'cuda' (or 'cpp').
        language = "c" if file_ext == "c" else "cuda"
        with open(filepath, "r", encoding="utf-8") as f:
            code_content = f.read()
        return f"```{language}\n{code_content}\n```"
    
def json_to_markdown(json_file, output_file, resources_dir="resources"):
    """
    Reads a JSON file containing quizzes, gathers resources from resources_dir,
    and converts everything to a Markdown file with formatted output.
    """
    # 1. Load the JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Gather resources from the directory
    all_resources = gather_resources(resources_dir)
    
    md_lines = []
    # 3. Process each quiz
    for quiz_name, questions in data.items():
        # Quiz name might be something like "Quiz1", so let's extract just the digit portion:
        # e.g., "Quiz1" -> "1". We'll handle if there's no digit gracefully.
        quiz_num_match = re.search(r"(\d+)$", quiz_name)
        quiz_num_str = quiz_num_match.group(1) if quiz_num_match else ""
        
        md_lines.append(f"# {quiz_name}")
        md_lines.append("")  # Blank line for spacing
        
        # 4. Process each question within the quiz
        for q_num, q_content in questions.items():
            md_lines.append(f"## Question {q_num}")
            md_lines.append("")
            
            # --- QUESTION SECTION ---
            question_text = q_content.get("question")
            md_lines.append(f"**Question:** {question_text}")
            md_lines.append("")
            
            # Insert resources for the "question" section if any
            if quiz_num_str in all_resources and q_num in all_resources[quiz_num_str]:
                if "question" in all_resources[quiz_num_str][q_num]:
                    for (res_path, res_ext) in all_resources[quiz_num_str][q_num]["question"]:
                        md_lines.append(embed_resource(res_path, res_ext))
                        md_lines.append("")
            
            # --- OPTIONS SECTION ---
            options = q_content.get("options", {})
            if options:
                md_lines.append("**Options:**")
                for option_label, option_text in options.items():
                    md_lines.append(f"- **{option_label}:** {option_text}")
                    
                    # Check if there are resources specifically for "options-<option_label>"
                    section_name = f"options-{option_label}"
                    if quiz_num_str in all_resources and q_num in all_resources[quiz_num_str]:
                        if section_name in all_resources[quiz_num_str][q_num]:
                            for (res_path, res_ext) in all_resources[quiz_num_str][q_num][section_name]:
                                md_lines.append(embed_resource(res_path, res_ext))
                                md_lines.append("")
                
                md_lines.append("")
            
            # --- ANSWER SECTION ---
            answer = q_content.get("answer", "")
            if answer:
                md_lines.append(f"**Answer:** {answer}")
                md_lines.append("")
                
                # Insert resources for the "answer" section if any
                if quiz_num_str in all_resources and q_num in all_resources[quiz_num_str]:
                    if "answer" in all_resources[quiz_num_str][q_num]:
                        for (res_path, res_ext) in all_resources[quiz_num_str][q_num]["answer"]:
                            md_lines.append(embed_resource(res_path, res_ext))
                            md_lines.append("")
            
            # --- PROFESSOR EXPLANATION SECTION ---
            professor_explanation = q_content.get("professor_explanation", "").strip()
            if professor_explanation:
                md_lines.append(f"**Professor Explanation:** {professor_explanation}")
                md_lines.append("")
                
                # Insert resources for "professor_explanation" if any
                if quiz_num_str in all_resources and q_num in all_resources[quiz_num_str]:
                    if "professor_explanation" in all_resources[quiz_num_str][q_num]:
                        for (res_path, res_ext) in all_resources[quiz_num_str][q_num]["professor_explanation"]:
                            md_lines.append(embed_resource(res_path, res_ext))
                            md_lines.append("")
            
            # --- ADDITIONAL EXPLANATION SECTION ---
            explanation = q_content.get("explanation", "").strip()
            if explanation:
                md_lines.append(f"**Explanation:** {explanation}")
                md_lines.append("")
                
                # Insert resources for "explanation" if any
                if quiz_num_str in all_resources and q_num in all_resources[quiz_num_str]:
                    if "explanation" in all_resources[quiz_num_str][q_num]:
                        for (res_path, res_ext) in all_resources[quiz_num_str][q_num]["explanation"]:
                            md_lines.append(embed_resource(res_path, res_ext))
                            md_lines.append("")
            
            # Separator between questions
            md_lines.append("---")
            md_lines.append("")
    
    # 5. Write out the markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    # Example usage (adjust file paths as needed):
    json_file = "questions.json"
    output_file = "quiz.md"
    resources_dir = "resources"
    json_to_markdown(json_file, output_file, resources_dir)