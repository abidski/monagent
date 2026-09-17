import os


# --- Agent ---
os.environ["GROQ_API_KEY"] = input("Enter Groq API key: ")
# --- Test ---
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What kind of scan is this: {image_path}",
                }
            ]
        }
    )
    print(f"\n{result['messages'][-1].content}\n")
