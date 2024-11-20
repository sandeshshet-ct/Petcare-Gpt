from flask import Flask, render_template, request, jsonify
import openai
import speech_recognition as sr
import pyttsx3
from deep_translator import GoogleTranslator
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import logging
from dotenv import load_dotenv
import os
# from Bio import Entrez

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure API keys
OPENAI_API_KEY =  ''
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# # Set up Qdrant client
# qdrant_client = QdrantClient(":memory:")  # Use ":memory:" for local in-memory storage
# embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vectorstore = Qdrant(client=qdrant_client, collection_name="chat_history", embeddings=embedding_model)


# Import and configure Google's generative AI after installing
try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    raise

# Initialize API clients
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Set up environment variables as a backup
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY


class ChatBot:
    def __init__(self):
        self.input_prompt = "Welcome to the Pet Care ChatBot. Ask any questions related to pet care and pets!"
        self.memory = ConversationBufferMemory()
        self.openai_llm = ChatOpenAI(api_key=OPENAI_API_KEY)
        self.conversation_chain = ConversationChain(llm=self.openai_llm, memory=self.memory)
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        logging.info("Text-to-speech engine initialized successfully.")

    def translate_text(self, text, target_language):
        try:
            translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
            return translated_text
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text

    # def fetch_papers(pmids):
    #     """
    #     Fetch details of papers given a list of PMIDs.
    #     """
    #     papers = []
    #     try:
    #         handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    #         records = Entrez.read(handle)
    #         handle.close()
    #
    #         for record in records['PubmedArticle']:
    #             paper_info = {}
    #             paper_info['pmid'] = record['MedlineCitation']['PMID']
    #             pmc_ids = [id_elem for id_elem in record['PubmedData']['ArticleIdList'] if
    #                        id_elem.attributes['IdType'] == 'pmc']
    #             paper_info['pmc'] = pmc_ids[0] if pmc_ids else 'N/A'
    #             paper_info['title'] = record['MedlineCitation']['Article']['ArticleTitle']
    #             paper_info['abstract'] = record['MedlineCitation']['Article']['Abstract']['AbstractText'][
    #                 0] if 'Abstract' in record['MedlineCitation']['Article'] else 'N/A'
    #             paper_info['keywords'] = [kw['DescriptorName'] for kw in
    #                                       record['MedlineCitation'].get('MeshHeadingList', [])]
    #             papers.append(paper_info)
    #     except Exception as e:
    #         print(f"An error occurred while fetching papers: {e}")
    #
    #     return papers

    def chat(self, msg, target_lang='en'):
        try:
            logging.debug(f"Original message: {msg}")
            if target_lang != 'en':
                msg = self.translate_text(msg, 'en')
                logging.debug(f"Translated message to English: {msg}")

            response = self.conversation_chain.invoke(input=self.input_prompt + "\n" + msg)
            logging.debug(f"Response from OpenAI: {response}")

            if target_lang != 'en':
                response['response'] = self.translate_text(response['response'], target_lang)
                logging.debug(f"Translated response to target language: {response['response']}")

            return jsonify(response)
        except Exception as e:
            logging.error(f"Chat error: {e}")
            return jsonify({"error": str(e)})

    def voice_to_text(self):
        with sr.Microphone() as source:
            logging.info("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                logging.info(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                logging.error("Could not understand the audio")
            except sr.RequestError as e:
                logging.error(f"Error with Google Speech Recognition service; {e}")
        return None

    def text_to_speech(self, text):
        try:
            logging.info(f"Converting text to speech: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"Text-to-speech error: {e}")


# Initialize ChatBot instance
chatbot = ChatBot()


def get_gemini_response(input_text, image_parts, prompt):
    # Update to use the new model, 'gemini-1.5-flash' instead of the deprecated 'gemini-pro-vision'
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Generate content with the new model
    response = model.generate_content([input_text, image_parts[0], prompt])

    # Return the generated response text
    return response.text


def input_image_setup(file_storage):
    if file_storage is not None:
        bytes_data = file_storage.read()
        image_parts = [
            {
                "mime_type": file_storage.content_type,
                "data": bytes_data
            }
        ]
        return image_parts
    raise FileNotFoundError("No file uploaded")


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        target_lang = request.form.get("lang", "en")
        response = chatbot.chat(msg, target_lang)
        return response
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/voice", methods=["POST"])
def voice():
    text = chatbot.voice_to_text()
    if text:
        target_lang = request.form.get("lang", "en")
        response = chatbot.chat(text, target_lang)
        chatbot.text_to_speech(response.get_json()["response"])
        return response
    return jsonify({"response": "Could not understand audio."})


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_interface():
    try:
        if request.method == 'POST':
            if request.form['submit_button'] == 'Text Input':
                msg = request.form['msg']
                target_lang = request.form.get('lang', 'en')
                response = chatbot.chat(msg, target_lang)
                return render_template('chat.html', response=response)
            elif request.form['submit_button'] == 'Voice Input':
                return render_template('voice.html')
        else:
            return render_template('chat.html')
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        input_text = request.form['input']
        uploaded_file = request.files['file']

        if uploaded_file:
            image_parts = input_image_setup(uploaded_file)
            prompt = """
            You are an expert in health management where you need to identify the symptoms of pet diseases from the provided description or image and provide detailed information on the disease, its causes, prevention measures, and recommend appropriate medications if necessary.

            Your response should be in the following format:

            1. Disease Name:
               - Symptoms:
               - Causes:
               - Prevention Measures:
               - Recommended Medications (if applicable):

            Please provide comprehensive information to assist users in understanding and managing their health effectively. You should not answer questions other than health topics, and you should mention a disclaimer at the end of the answers/context that you are not an expert. Please ensure to connect with a health professional.
            """
            response = get_gemini_response(input_text, image_parts, prompt)

            target_lang = request.form.get("lang", "en")
            if target_lang != 'en':
                response = chatbot.translate_text(response, target_lang)

            return render_template('disease.html', response=response)

    return render_template('disease.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000, threaded=False)
