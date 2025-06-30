import gradio as gr

# def greet(name):
#     return f"Hello, {name}!"
#
# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()
def analyze(name, age):
    return f"{name} is {age} years old."

demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Textbox(label="Name"), gr.Number(label="Age")],
    outputs="text"
)
demo.launch(pwa=True)
