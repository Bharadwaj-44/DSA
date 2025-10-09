import shutil
import gradio as gr
import json
import time
from conversation import Conversation
from prompt_engineering.prompts import *
import yaml
from utils.utils import *
import sys
import os


class DSA:
    def __init__(self, config_path='config.yaml'):
        print("Try to load config: ", config_path)

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = os.path.dirname(sys.executable)
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(bundle_dir, config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        if self.config["load_chat"] == True:
            self.load_dialogue(self.config["chat_history_path"])
        else:
            self.session_cache_path = self.init_local_cache_path(to_absolute_path(self.config["project_cache_path"]))
            self.config["session_cache_path"] = self.session_cache_path
        print("Session cache path: ", self.session_cache_path)
        self.conv = Conversation(self.config)

        self.conv.programmer.messages = [
            {
                "role": "system",
                "content": PROGRAMMER_PROMPT.format(working_path=self.session_cache_path)
            }
        ]

        if self.conv.retrieval:
            self.conv.programmer.messages[0]["content"] += KNOWLEDGE_INTEGRATION_SYSTEM


    def init_local_cache_path(self, project_cache_path):
        current_fold = time.strftime('%Y-%m-%d', time.localtime())
        hsid = str(hash(id(self)))  # new_uuid = str(uuid.uuid4())
        session_cache_path = os.path.join(project_cache_path, current_fold + '-' + hsid)
        if not os.path.exists(session_cache_path):
            os.makedirs(session_cache_path)
        return session_cache_path

    def open_board(self):
        data = self.conv.show_data()
        if data.empty:
            print("No data available to display")
            return gr.Dataframe(visible=False)
        else:
            print(f"Displaying dataframe with {len(data)} rows and {len(data.columns)} columns")
            return gr.Dataframe(value=data, visible=True)

    def add_file(self, files):

        file_path = files.name
    
        shutil.copy(file_path, self.session_cache_path)
    
        filename = os.path.basename(file_path)
    
        self.conv.add_data(file_path)
    
        self.conv.file_list.append(filename)
    
        local_cache_path = os.path.join(self.session_cache_path, filename)
    
        # Get dataset description with error handling and truncation
    
        try:
        
            gen_info = self.conv.my_data_cache.get_description()
    
            gen_info_str = str(gen_info)
    
            # Truncate if too long (limit to 3000 characters for large datasets)
    
            max_length = 3000
    
            if len(gen_info_str) > max_length:
            
                gen_info_str = gen_info_str[:max_length] + f"\n... (dataset info truncated from {len(gen_info_str)} to {max_length} characters. Full data is still available for analysis.)"
    
                print(f"Dataset info truncated: {len(str(gen_info))} → {max_length} characters")
    
            dataset_context = f"\nNow, user uploads the data in {local_cache_path}\n, and here is the general information of the dataset:\n{gen_info_str}\nYou should care about the missing values and type of each column in your later processing."
    
        except Exception as e:
        
            print(f"Warning: Could not get dataset description: {e}")
    
            import traceback
    
            traceback.print_exc()
    
            dataset_context = f"\nNow, user uploads the data in {local_cache_path}. Dataset loaded successfully. You can use pd.read_csv('{local_cache_path}') to analyze it."
    
        self.conv.programmer.messages[0]["content"] += dataset_context
    
        print(f"Upload file in gradio path: {file_path}, local cache path: {local_cache_path}")
 
    def add_file_with_feedback(self, files):
        """Add file with status feedback"""
        if files is None:
            return gr.HTML(visible=False)
        
        try:
            file_path = files.name
            filename = os.path.basename(file_path)
            
            # Copy file to session cache
            shutil.copy(file_path, self.session_cache_path)
            
            # Add data to conversation
            self.conv.add_data(file_path)
            self.conv.file_list.append(filename)
            
            # Get dataset information
            # Get dataset information with truncation

            local_cache_path = os.path.join(self.session_cache_path, filename)

            try:
            
                gen_info = self.conv.my_data_cache.get_description()

                gen_info_str = str(gen_info) if gen_info else "Dataset information not available"

                # Truncate if too long

                max_length = 3000

                if len(gen_info_str) > max_length:
                
                    gen_info_str = gen_info_str[:max_length] + f"\n... (dataset info truncated from {len(gen_info_str)} to {max_length} characters. Full data is still available for analysis.)"

                    print(f"Dataset info truncated: {len(str(gen_info))} → {max_length} characters")

            except Exception as e:
            
                print(f"Warning: Could not get dataset description: {e}")

                import traceback

                traceback.print_exc()

                gen_info_str = f"Dataset loaded successfully (info generation failed: {str(e)})"

            # Update programmer messages

            dataset_context = f"\nNow, user uploads the data in {local_cache_path}\n, and here is the general information of the dataset:\n{gen_info_str}\nYou should care about the missing values and type of each column in your later processing."

            self.conv.programmer.messages[0]["content"] += dataset_context
 
            # ADD THESE DEBUG LINES:
            print("=" * 80)
            print("SYSTEM MESSAGE AFTER FILE UPLOAD:")
            print(self.conv.programmer.messages[0]["content"])
            print("=" * 80)
 
            # Create success status HTML
            status_html = f"""
            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>✅ File Uploaded Successfully!</strong><br>
                <strong>File:</strong> {filename}<br>
                <strong>Size:</strong> {os.path.getsize(file_path):,} bytes<br>
                <strong>Type:</strong> {filename.split('.')[-1].upper()}
            </div>
            """
            
            print(f"Upload file in gradio path: {file_path}, local cache path: {local_cache_path}")
            
            # Return only the status HTML, no chat message
            return gr.HTML(value=status_html, visible=True)
            
        except Exception as e:
            # Create error status HTML
            error_html = f"""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>❌ Upload Failed!</strong><br>
                <strong>Error:</strong> {str(e)}
            </div>
            """
            
            # Return only the error status HTML, no chat message
            return gr.HTML(value=error_html, visible=True)

    def rendering_code(self):
        return self.conv.rendering_code()

    def generate_report(self, chat_history):
        print(f"DEBUG: generate_report called with chat_history length: {len(chat_history) if chat_history else 0}")
        try:
            down_path = self.conv.document_generation(chat_history)
            print(f"DEBUG: Report generated at path: {down_path}")
            # Convert to absolute path and ensure it exists
            abs_path = os.path.abspath(down_path)
            if os.path.exists(abs_path):
                print(f"DEBUG: File exists at: {abs_path}")
                return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Report", value=abs_path, visible=True)]
            else:
                print(f"ERROR: File not found at: {abs_path}")
                return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
        except Exception as e:
            print(f"ERROR in generate_report: {e}")
            import traceback
            traceback.print_exc()
            # Return error state
            return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    def export_code(self):
        down_path = self.conv.export_code()
        # Convert to absolute path and ensure it exists
        abs_path = os.path.abspath(down_path)
        if os.path.exists(abs_path):
            print(f"DEBUG: Notebook file exists at: {abs_path}")
            return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Notebook", value=abs_path, visible=True)]
        else:
            print(f"ERROR: Notebook file not found at: {abs_path}")
            return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    def down_report(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    def down_notebook(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
    
    
    def chat_streaming(self, message, chat_history, code=None):
        if not code:
            self.conv.programmer.messages.append({"role": "user", "content": message})
        else:
            message = code
        return "", chat_history + [[message, None]]
    
    def show_csv_download(self, filename):
        """Show CSV download link when specifically requested"""
        # Check in session cache first
        cache_path = os.path.join(self.session_cache_path, filename)
        if os.path.exists(cache_path):
            return f"📥 **{filename}** is available for download. Use the download buttons above to get the file."
        
        # Check in root directory
        root_path = os.path.join(os.path.dirname(self.session_cache_path), filename)
        if os.path.exists(root_path):
            return f"📥 **{filename}** is available for download. Use the download buttons above to get the file."
        
        return f"❌ {filename} not found"
    
    
    def get_csv_file_path(self, filename):
        """Get CSV file path for download"""
        # Check in session cache first
        cache_path = os.path.join(self.session_cache_path, filename)
        if os.path.exists(cache_path):
            return os.path.abspath(cache_path)
        
        # Check in root directory
        root_path = os.path.join(os.path.dirname(self.session_cache_path), filename)
        if os.path.exists(root_path):
            return os.path.abspath(root_path)
        
        return None
    
    def get_csv_download_path(self):
        """Get CSV download path for Gradio DownloadButton"""
        # Try sample_data.csv first
        sample_path = self.get_csv_file_path("sample_data.csv")
        if sample_path:
            return sample_path
        
        # Try test_dataset.csv
        test_path = self.get_csv_file_path("test_dataset.csv")
        if test_path:
            return test_path
            
        # Try insurance.csv
        insurance_path = self.get_csv_file_path("insurance.csv")
        if insurance_path:
            return insurance_path
            
        return None
    
    def get_download_path(self):
        """Get CSV file for download"""
        # Only handle CSV files now
        csv_path = self.get_csv_download_path()
        if csv_path:
            return csv_path
            
        return None
    
    def download_file(self):
        """Download file function for Gradio DownloadButton"""
        file_path = self.get_download_path()
        if file_path and os.path.exists(file_path):
            return file_path
        return None
    
    def show_csv_download_button(self):
        """Show CSV download button when files are available"""
        csv_path = self.get_csv_download_path()
        if csv_path:
            return [gr.DownloadButton("Download CSV", value=csv_path, visible=True)]
        else:
            return [gr.DownloadButton("Download CSV", visible=False)]
    
    
    

    def save_dialogue(self, chat_history):
        self.conv.save_conv()
        with open(os.path.join(self.session_cache_path, 'system_dialogue.json'), 'w') as f:
            json.dump(chat_history, f, indent=4)
        print(f"Dialogue saved in {os.path.join(self.session_cache_path, 'system_dialogue.json')}.")

    def load_dialogue(self, dialogue_path):
        try:
            system_dialogue_path = os.path.join(dialogue_path, 'system_dialogue.json')
            system_config_path = os.path.join(dialogue_path, 'config.json')
            with open(system_dialogue_path, 'r') as f:
                chat_history = json.load(f)
            with open(system_config_path, 'r') as f:
                sys_config = json.load(f)
            self.session_cache_path = sys_config["session_cache_path"]
            self.config["session_cache_path"] = self.session_cache_path
            self.config["chat_history_display"] = chat_history
            self.config["figure_list"] = sys_config["figure_list"]
            return chat_history
        except Exception as e:
            print(f"Failed to load the chat history: {e}")
            return []

    def clear_all(self, message, chat_history):
        self.conv.clear()
        return "", []

    def update_config(self, conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time,
                      load_chat, chat_history_path):

        self.conv.update_config(conv_model=conv_model, programmer_model=programmer_model, inspector_model=inspector_model, api_key=api_key,
                      base_url_conv_model=base_url_conv_model, base_url_programmer=base_url_programmer, base_url_inspector=base_url_inspector,
                      max_attempts=max_attempts, max_exe_time=max_exe_time)

        if load_chat == True:
            self.config['chat_history_path'] = chat_history_path
            chat_history = self.load_dialogue(chat_history_path)
            self.config['load_chat'] = load_chat
            return ["### Config Updated!", chat_history]

        return "### Config Updated!", []
    def debug_system_message(self):

        """Debug function to check system message"""
    
        print("\n" + "=" * 80)
    
        print("DEBUG: SYSTEM MESSAGE CHECK")
    
        print("=" * 80)
    
        if len(self.conv.programmer.messages) > 0:
        
            sys_msg = self.conv.programmer.messages[0]
    
            print(f"Role: {sys_msg.get('role')}")
    
            print(f"Content length: {len(sys_msg.get('content', ''))}")
    
            print(f"Content preview (first 500 chars):")
    
            print(sys_msg.get('content', '')[:500])
    
            print(f"\n...Content preview (last 500 chars):")
    
            print(sys_msg.get('content', '')[-500:])
    
        else:
        
            print("ERROR: No system message found!")
    
        print("=" * 80 + "\n")
 
