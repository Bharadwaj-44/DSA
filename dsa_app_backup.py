import sys
import asyncio
 
# Fix for Python 3.12 + uvicorn + Gradio compatibility
if sys.platform == 'win32' and sys.version_info >= (3, 12):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
 
import os
import time
print("🚀 Starting DSA Application...")
start_time = time.time()
# Set matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'Agg'
print("📊 Loading matplotlib...")
# ... rest continues as before
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
 
print("🎨 Loading Gradio...")
import gradio as gr
print("🖼️ Loading frontend components...")
# Commented out to test if these cause issues
# from front_end.js import js
# from front_end.css import css
print("📦 Loading DSA core...")
from DSA import DSA
from utils.utils import to_absolute_path
 
# Additional matplotlib configuration for Gradio compatibility
print("🔧 Configuring matplotlib...")
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
 
print(f"✅ Libraries loaded in {time.time() - start_time:.2f} seconds")
 
def launch_app():
    print("🔵 DEBUG 1: Starting launch_app() function")
    
    print("🔵 DEBUG 2: Creating DSA instance...")
    dsa = DSA(config_path='config.yaml')
    print("🔵 DEBUG 3: DSA instance created successfully")
    
    print("🔵 DEBUG 4: About to create Gradio Blocks...")
    # Removed css=css, js=js to test without them
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        print("🔵 DEBUG 5: Inside gr.Blocks context")

        print("🔵 DEBUG 6: Creating Tab...")
        with gr.Tab("Data Science Agent"):
            print("🔵 DEBUG 7: Inside Tab context")
            
            print("🔵 DEBUG 8: Creating HTML header...")
            gr.HTML("<H1>Welcome to Data Science Agent! Easy Data Analysis!</H1>")
            print("🔵 DEBUG 9: HTML created")
            
            print("🔵 DEBUG 10: Creating upload status...")
            upload_status = gr.HTML(value="", visible=False)
            print("🔵 DEBUG 11: Upload status created")
            
            print("🔵 DEBUG 12: Creating chatbot...")
            chatbot = gr.Chatbot(value=dsa.conv.chat_history_display, height=600, label="Data Science Agent", show_copy_button=True, type="tuples", render_markdown=True)
            print("🔵 DEBUG 13: Chatbot created")
            
            print("🔵 DEBUG 14: Creating first Group...")
            with gr.Group():
                print("🔵 DEBUG 15: Inside first Group")
                with gr.Row(equal_height=True):
                    print("🔵 DEBUG 16: Creating upload button...")
                    upload_btn = gr.UploadButton(label="Upload Data", file_types=[".csv", ".xlsx"], scale=1)
                    print("🔵 DEBUG 17: Upload button created")
                    
                    print("🔵 DEBUG 18: Creating message textbox...")
                    msg = gr.Textbox(show_label=False, placeholder="Sent message to LLM", scale=6, elem_id="chatbot_input")
                    print("🔵 DEBUG 19: Message textbox created")
                    
                    print("🔵 DEBUG 20: Creating submit button...")
                    submit = gr.Button("Submit", scale=1)
                    print("🔵 DEBUG 21: Submit button created")
            
            print("🔵 DEBUG 22: Creating second Row...")
            with gr.Row(equal_height=True):
                print("🔵 DEBUG 23: Creating action buttons...")
                board = gr.Button(value="Show/Update DataFrame", elem_id="df_btn", elem_classes="df_btn")
                export_notebook = gr.Button(value="Notebook")
                down_notebook = gr.DownloadButton("Download Notebook", visible=False)
                generate_report = gr.Button(value="Generate Report")
                down_report = gr.DownloadButton("Download Report", visible=False)
                download_csv = gr.DownloadButton("Download CSV", visible=False)
                edit = gr.Button(value="Edit Code", elem_id="ed_btn", elem_classes="ed_btn")
                save = gr.Button(value="Save Dialogue")
                clear = gr.ClearButton(value="Clear All")
                print("🔵 DEBUG 24: All action buttons created")
 
            print("🔵 DEBUG 25: Creating code editor group...")
            with gr.Group():
                with gr.Row(visible=False, elem_id="ed", elem_classes="ed"):
                    code = gr.Code(label="Code", scale=6)
                    code_btn = gr.Button("Submit Code", scale=1)
            print("🔵 DEBUG 26: Code editor created")
            
            print("🔵 DEBUG 27: Setting up code button click handler...")
            code_btn.click(fn=dsa.chat_streaming, inputs=[msg, chatbot, code], outputs=[msg, chatbot]).then(
                dsa.conv.stream_workflow, inputs=[chatbot, code], outputs=chatbot)
            print("🔵 DEBUG 28: Code button handler set")
 
            print("🔵 DEBUG 29: Creating dataframe...")
            df = gr.Dataframe(visible=False, elem_id="df", elem_classes="df")
            print("🔵 DEBUG 30: Dataframe created")
 
            def clear_upload_status():
                return gr.HTML(visible=False)
           
            print("🔵 DEBUG 31: Setting up event handlers...")
            upload_btn.upload(fn=clear_upload_status, outputs=upload_status).then(
                fn=dsa.add_file_with_feedback, inputs=upload_btn, outputs=[upload_status]
            )
            print("🔵 DEBUG 32: Upload handler set")
            
            msg.submit(dsa.chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                dsa.conv.stream_workflow, chatbot, chatbot
            )
            print("🔵 DEBUG 33: Message submit handler set")
            
            submit.click(dsa.chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                dsa.conv.stream_workflow, chatbot, chatbot
            )
            print("🔵 DEBUG 34: Submit click handler set")
            
            board.click(dsa.open_board, inputs=[], outputs=df)
            edit.click(dsa.rendering_code, inputs=None, outputs=code)
            export_notebook.click(dsa.export_code, inputs=None, outputs=[export_notebook, down_notebook])
            down_notebook.click(dsa.down_notebook, inputs=None, outputs=[export_notebook, down_notebook])
            generate_report.click(dsa.generate_report, inputs=[chatbot], outputs=[generate_report, down_report])
            down_report.click(dsa.down_report, inputs=None, outputs=[generate_report, down_report])
            download_csv.click(fn=dsa.download_file, outputs=download_csv)
            save.click(dsa.save_dialogue, inputs=chatbot)
            clear.click(fn=dsa.clear_all, inputs=[msg, chatbot], outputs=[msg, chatbot])
            print("🔵 DEBUG 35: All event handlers set")
 
        # The Configuration Page
        print("🔵 DEBUG 36: Creating Configuration tab...")
        with gr.Tab("Configuration"):
            gr.Markdown("# System Configuration for Data Science Agent")
            with gr.Row():
                conv_model = gr.Textbox(value="gpt-3.5-turbo", label="Conversation Model")
                programmer_model = gr.Textbox(value="gpt-3.5-turbo", label="Programmer Model")
                inspector_model = gr.Textbox(value="gpt-3.5-turbo", label="Inspector Model")
           
            api_key = gr.Textbox(label="API Key", type="password", placeholder="Input Your API key")
            with gr.Row():
                base_url_conv_model = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Conv Model)")
                base_url_programmer = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Programmer)")
                base_url_inspector = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Inspector)")
 
            with gr.Row():
                max_attempts = gr.Number(value=5, label="Max Attempts", precision=0)
                max_exe_time = gr.Number(value=18000, label="Max Execution Time (s)", precision=0)
            with gr.Row():            
                load_chat = gr.Checkbox(value=False, label="Load from Cache")
                chat_history_path = gr.Textbox(label="Chat History Path", visible=False, interactive=True)
               
            save_btn = gr.Button("Save Configuration", variant="primary")
            status_output = gr.Markdown("")
           
            def toggle_chat_history_path(load_chat_checked):
                return gr.Textbox(visible=load_chat_checked, interactive=True)
           
            save_btn.click(
                fn=dsa.update_config,
                inputs=[
                    conv_model, programmer_model, inspector_model, api_key,
                    base_url_conv_model, base_url_programmer, base_url_inspector,
                    max_attempts, max_exe_time,
                    load_chat, chat_history_path
                ],
                outputs=[status_output, chatbot]
            )
 
            load_chat.change(
                fn=toggle_chat_history_path,
                inputs=load_chat,
                outputs=chat_history_path
            )
        print("🔵 DEBUG 37: Configuration tab created")
 
    print("🔵 DEBUG 38: Exited gr.Blocks context")
    
    # Get all possible cache paths
    print("🔵 DEBUG 39: Setting up allowed paths...")
    allowed_paths = [
        to_absolute_path(dsa.config["project_cache_path"]),
        to_absolute_path("cache"),
        to_absolute_path(dsa.session_cache_path),
        dsa.session_cache_path,  # Direct session cache path
        os.path.dirname(dsa.session_cache_path),  # Parent directory
        to_absolute_path(".")  # Project root
    ]
    print("🔵 DEBUG 40: Allowed paths set")
   
    print("🔵 DEBUG 41: About to call demo.launch()...")
    print("=" * 60)
    print("🚀 LAUNCHING GRADIO SERVER...")
    print("=" * 60)
    
    # ✅ FIXED SETTINGS FOR EC2
    demo.launch(
        server_name="0.0.0.0",    # ✅ Listen on all interfaces (not 127.0.0.1)
        server_port=7860,         # ✅ Explicit port
        allowed_paths=allowed_paths,
        share=False,              # ✅ No public tunnel (not True)
        inbrowser=False           # ✅ Don't try to open browser (not True)
    )
    
    print("🔵 DEBUG 42: demo.launch() completed")
 
 
if __name__ == '__main__':
    print("=" * 60)
    print("🎬 MAIN ENTRY POINT")
    print("=" * 60)
    launch_app()
