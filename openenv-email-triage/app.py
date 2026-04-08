import gradio as gr
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# TASKS
# ---------------------------
tasks = [
    "Urgent: Server is down!",
    "Meeting scheduled tomorrow",
    "You won a lottery!!!",
    "Client complaint about service",
    "Invoice pending payment"
]

state = {
    "index": 0,
    "score": 0
}

# ---------------------------
# AI CLASSIFIER (IMPROVED)
# ---------------------------
def ai_classify(email):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert email triage assistant.\n"
                        "Classify the email into ONE of these actions: Respond, Ignore, Escalate, Forward.\n"
                        "Also give a confidence score (0-100) and short explanation.\n\n"
                        "Return ONLY in this format:\n"
                        "Action:<action>\n"
                        "Confidence:<number>\n"
                        "Explanation:<reason>"
                    )
                },
                {"role": "user", "content": email}
            ]
        )

        result = response.choices[0].message.content

        # Parse output
        action, confidence, explanation = "N/A", "N/A", "N/A"

        for line in result.split("\n"):
            if "Action:" in line:
                action = line.split(":")[1].strip()
            elif "Confidence:" in line:
                confidence = line.split(":")[1].strip() + "%"
            elif "Explanation:" in line:
                explanation = line.split(":")[1].strip()

        return action, confidence, explanation

    except Exception as e:
        return "Error", "0%", str(e)

# ---------------------------
# GAME LOGIC
# ---------------------------
def get_email():
    if state["index"] >= len(tasks):
        return "🎉 Done!", "Final Score: " + str(state["score"]), ""
    return tasks[state["index"]], "", f"Step {state['index']+1}/{len(tasks)}"

def take_action(action):
    email = tasks[state["index"]]

    correct_map = {
        "urgent": "Escalate",
        "meeting": "Respond",
        "lottery": "Ignore",
        "complaint": "Escalate",
        "invoice": "Forward"
    }

    correct = "Respond"
    for key in correct_map:
        if key in email.lower():
            correct = correct_map[key]

    if action == correct:
        state["score"] += 1
        feedback = "✅ Correct"
    else:
        feedback = f"❌ Wrong (Correct: {correct})"

    state["index"] += 1

    if state["index"] < len(tasks):
        next_email = tasks[state["index"]]
    else:
        next_email = "🎉 No more emails!"

    progress = f"Step {min(state['index']+1, len(tasks))}/{len(tasks)}"

    return next_email, f"{feedback} | Score: {state['score']}", progress

# ---------------------------
# LEADERBOARD
# ---------------------------
leaderboard = []

def save_score(name):
    leaderboard.append((name, state["score"]))
    return sorted(leaderboard, key=lambda x: x[1], reverse=True)

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🚀 AI-Powered Email Triage System")
    gr.Markdown("### OpenEnv Simulation + AI Decision + Explainability")

    gr.Markdown("""
    ### 🧠 How to use:
    1. Read the email  
    2. Choose correct action  
    3. Use AI Suggestion for help  
    4. Track your score  
    """)

    email_box = gr.Textbox(label="📩 Current Email", interactive=False)
    feedback = gr.Textbox(label="📊 Feedback")
    progress = gr.Textbox(label="📈 Progress")

    gr.Markdown("### 🎯 Actions")
    with gr.Row():
        gr.Button("Respond").click(lambda: take_action("Respond"), outputs=[email_box, feedback, progress])
        gr.Button("Ignore").click(lambda: take_action("Ignore"), outputs=[email_box, feedback, progress])
        gr.Button("Escalate").click(lambda: take_action("Escalate"), outputs=[email_box, feedback, progress])
        gr.Button("Forward").click(lambda: take_action("Forward"), outputs=[email_box, feedback, progress])

    gr.Markdown("### 🤖 AI Suggestion")

    ai_action = gr.Textbox(label="Action")
    ai_confidence = gr.Textbox(label="Confidence")
    ai_explanation = gr.Textbox(label="Explanation", lines=3)

    gr.Button("Get AI Suggestion").click(
        ai_classify,
        inputs=email_box,
        outputs=[ai_action, ai_confidence, ai_explanation]
    )

    gr.Markdown("### 🏆 Leaderboard")
    name_input = gr.Textbox(label="Enter your name")
    leaderboard_output = gr.Dataframe(headers=["Name", "Score"])

    gr.Button("Save Score").click(save_score, inputs=name_input, outputs=leaderboard_output)

    demo.load(get_email, outputs=[email_box, feedback, progress])

demo.launch()
