import streamlit as st
from PIL import Image as PIL_Image
import http.client
import typing
import urllib.request
import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from io import BytesIO
#from llmsherpa.readers import LayoutPDFReader
#from llmsherpa.readers.layout_reader import Document
#import google.generativeai as genai
#from vertexai.preview.generative_models import Part
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
    HarmCategory,
    HarmBlockThreshold,
)

def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes

def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)

def convert_pdf_to_images(file_path, scale=300/72):
    
    pdf_file = pdfium.PdfDocument(file_path)  
    page_indices = [i for i in range(len(pdf_file))]
    
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices, 
        scale = scale,
    )
    
    list_final_images = [] 
    
    for i, image in zip(page_indices, renderer):
        
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i:image_byte_array}))
    
    return list_final_images

def display_images(list_dict_final_images):
    
    all_images = [list(data.values())[0] for data in list_dict_final_images]

    for index, image_bytes in enumerate(all_images):

        image = PIL_Image.open(BytesIO(image_bytes))
        figure = plt.figure(figsize = (image.width / 100, image.height / 100))

        plt.title(f"----- Page Number {index+1} -----")
        plt.imshow(image)
        plt.axis("off")
        plt.show()

def describe_images_gemini(list_dict_final_images,prompt):
    
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    
    for index, image_bytes in enumerate(image_list):
        print(f"Processing image number : {index}")
        #prompt = """You are an Bank assistant reading a bank statement, invoices and processing PDFs for Vattanac Bank in Cambodia.
        #Extract all the text from the image. Describe the image, understand the transactions, amount credited and debited, items in the invoices, their price explain in detail. If there are tables and flowcharts, summarize them. If the items are in khmer, translate and summarize in English.
        #"""
        image_data = Image.from_bytes(image_bytes)
        contents = [image_data, prompt]

        #config = GenerationConfig(max_output_tokens=2048,temperature=0.4)
        config = gen_config

        safety_config = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        try:    
            responses = GenerativeModel(option).generate_content(contents, generation_config=config, stream=False,safety_settings=safety_config)
        except ValueError as e:
            print(f"Something went wrong with the API call: {e}")
            # If the response doesn't contain text, check if the prompt was blocked.
            print(responses.prompt_feedback)
            # Also check the finish reason to see if the response was blocked.
            print(responses.candidates[0].finish_reason)
            # If the finish reason was SAFETY, the safety ratings have more details.
            print(responses.candidates[0].safety_ratings)
            raise Exception(f"Something went wrong with the API call: {e}")

        #messages = ""
        #for response in responses:
        #    print(response.text, end="")
    #     messages = messages + response.text
        msg=responses.text
        print(f"Enriched text: {msg}")
        image_content.append(msg)
    return "\n".join(image_content) 

#def load_pdf_from_url(pdf_url: str) -> Document:
#    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
#    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
#    doc = pdf_reader.read_pdf(pdf_url)
#    return doc

st.set_page_config(page_title="Gemini Pro Multimodal Processor")

st.write("Welcome to the Gemini Pro Multimodal Processor. You can proceed by providing your Google API Key")

#with st.expander("Provide Your Google API Key"):
#     google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
     
#if not google_api_key:
#    st.info("Enter the Google API Key to continue")
#    st.stop()

#genai.configure(api_key=google_api_key)

st.title("Gemini Pro Multimodal Processor")

with st.sidebar:
    option = st.selectbox('Choose Your Model',('gemini-pro', 'gemini-pro-vision'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option
    
    st.write("Adjust Your Parameter Here:")
    temperature = st.number_input("Temperature", min_value=0.0, max_value= 1.0, value =0.5, step =0.01)
    max_token = st.number_input("Maximum Output Token", min_value=0, value =1000)
    gen_config = GenerationConfig(max_output_tokens=max_token,temperature=temperature)
    
    st.divider()
    
    #upload_image = st.file_uploader("Upload Your Image Here", accept_multiple_files=False, type = ['jpg', 'png'])
    upload_image =  st.text_input("Enter Your Image URL Here")
    if upload_image:
        image = load_image_from_url(upload_image)
    st.divider()

    upload_pdf =  st.file_uploader("Upload Your PDF Here",type="pdf")
    if upload_pdf:
        image_dict = convert_pdf_to_images(upload_pdf)
    st.divider()


    upload_video =  st.text_input("Enter Your Video URL Here")
    if upload_video:
        video = Part.from_uri(
            uri=upload_video,
            mime_type="video/mp4",
        )
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_video:
    if option == "gemini-pro":
        st.info("Please Switch to the Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            #response=st.session_state.chat.send_message([prompt,video],stream=True,generation_config = gen_config)
            responses=GenerativeModel(option).generate_content([prompt,video],stream=True,generation_config = gen_config)
            #response.resolve()
            #msg = response.text
            messages = ""
            for response in responses:
                print(response.text, end="")
                messages = messages + response.text
            msg=messages    
            
            st.session_state.chat = GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})

            st.chat_message("assistant").write(msg)
elif upload_pdf:
    if option == "gemini-pro":
        st.info("Please Switch to the Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            text_with_gemini = describe_images_gemini(image_dict, prompt)
            st.session_state.chat = GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": text_with_gemini})
            #st.image(image_dict,width=300)
            st.chat_message("assistant").write(text_with_gemini)
elif upload_image:
    if option == "gemini-pro":
        st.info("Please Switch to the Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            #response=st.session_state.chat.send_message([prompt,image],stream=True,generation_config = gen_config)
            responses=GenerativeModel(option).generate_content([prompt,image],stream=True,generation_config = gen_config)
            #response.resolve()
            #msg=response.text
            messages = ""
            for response in responses:
                print(response.text, end="")
                messages = messages + response.text
            msg=messages  
            #print(responses.text)
            #msg=responses.text

            st.session_state.chat = GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            #st.image(image,width=300)
            st.chat_message("assistant").write(msg)

else:
    if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            #responses=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            responses=GenerativeModel(option).generate_content(prompt,stream=True,generation_config = gen_config)
            #response.resolve()
            #msg=response.text
            messages = ""
            for response in responses:
                print(response.text, end="")
                messages = messages + response.text
            msg=messages  
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    
    