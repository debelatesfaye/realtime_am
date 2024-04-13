from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from collections import deque
from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics

from default_turninator.src.data import Data
from default_turninator.src.turninator import Turninator 
from default_turninator.src.util import handle_errors
import logging
from default_segmenter.src.segmenter import Segmenter
from default_segmenter.src.data import Data
from default_segmenter.src.utility import handle_errors
from proposition_unitizer.src.data import Data, AIF
from proposition_unitizer.src.propositionalizer import  Propositionalizer 
from proposition_unitizer.src.utility import get_file,handle_errors
from decompose.get_components import FunctionalComponentsExtractor




from dialogpt_vanila.src.caasr import CAASRArgumentStructure
from transformers import GPT2Tokenizer,pipeline

from amf_fast_inference import model

from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import logging
import torch

logging.basicConfig(datefmt='%H:%M:%S', level=logging.DEBUG)

model_name = "debela-arg/dialogtp-am-medium"
loader = model.ModelLoader(model_name)
model = loader.load_model()        
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)




app = Flask(__name__)
socketio = SocketIO(app)

class TextProcessor:
    def __init__(self, ):
        self.inference_identifier_QUEUE = deque()
        self.segmenter_QUEUE = deque()
        self.total_segment_topic_dict = {}
        self.previous_map = None
        

    def segment_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            sentences = []
            for line in lines:
                # Split the line into sentences (assuming '.' as the sentence delimiter)
                sentences.extend(line.strip().split('.'))
            return sentences

    def turninator(self, xAIF):
        turninator = Turninator()
        turninator_result=turninator.turninator_default(xAIF)
        return turninator_result
    def propositionaliser(self, xAIF):
        propositionaliser = Propositionalizer()
        result=propositionaliser.propositionalizer_default(xAIF)
        return result

    def segmenter(self, xAIF):         
        segmenter = Segmenter()
        result=segmenter.segmenter_default(xAIF)
        return result

    def inference_identifier(self, xAIF):
        strcture_generator  = CAASRArgumentStructure(pipe,tokenizer)
        structure = strcture_generator.get_argument_structure(xAIF)
        return structure

    def merge_map(self, current_map, previous_map, linking_map):
            previous_map['aif']['nodes']+= current_map['aif']['nodes']
            previous_map['aif']['edges']+= current_map['aif']['edges']
            previous_map['aif']['locutions']+= current_map['aif']['locutions']
            previous_map['aif']['edges']+= linking_map['aif']['edges']

            return previous_map


    def get_topics(self, xAIF):
        segment_topic_dict = {}
        segmented_sentences = {}

        for node in xAIF['aif']['nodes']:
            if node['type'] == "I":
                segmented_sentences[node['text']] = node['nodeID']



        extractor = FunctionalComponentsExtractor()
        for nodeID, segment in segmented_sentences.items():
            merged_tc_c, merged_tc_p, merged_asp_c, merged_asp_p = extractor.get_model_based_functional_components((segment,segment))
            if merged_tc_c in segment_topic_dict:
                segment_topic_dict[merged_tc_c].append((nodeID, segment))
            else:
                segment_topic_dict[merged_tc_c] = [(nodeID, segment)]

            if merged_asp_c in segment_topic_dict:
                segment_topic_dict[merged_asp_c].append((nodeID, segment))
            else:
                segment_topic_dict[merged_asp_c] = [(nodeID, segment)]
            
        return segment_topic_dict

    def create_linking_map(self, segment_topic_dict):
        if not segment_topic_dict or self.total_segment_topic_dict:
            return {}, {}
        xAIF = {'aif':{'nodes':[]}}
        for topic,nodeID_segment in segment_topic_dict.items():
            if topic in self.total_segment_topic_dict: 
                    nodeID_segment2 = self.total_segment_topic_dict[topic]
                    for nodeID, segment in nodeID_segment:                       
                            entry = {'type':"I", "text":segment, "nodeID":nodeID}
                            xAIF['aif']['nodes'].append(entry)
                    for nodeID, segment in nodeID_segment2:                       
                            entry = {'type':"I", "text":segment, "nodeID":nodeID}
                            xAIF['aif']['nodes'].append(entry)

        for topic,nodeID_segment in segment_topic_dict.items():
            if topic in self.total_segment_topic_dict: 
                self.total_segment_topic_dict[topic].append(nodeID_segment)
            else:
                self.total_segment_topic_dict[topic] = [nodeID_segment]
                    
       
        return xAIF, self.total_segment_topic_dict

    def process_file(self, file_path):
        sentences = self.segment_file(file_path)
        for i in range(0, len(sentences), 2):
            batch = sentences[i:i + 2]

            # Pass batch to turninator
            turninator_output = self.turninator(batch)

            # Pass turninator_output to segmenter
            segmenter_output = self.segmenter(turninator_output)
            segment_topic_dict = self.get_topics(segmenter_output)
            inter_map_segments, self.total_segment_topic_dict = self.create_linking_map(segment_topic_dict)

            # Pass segmenter_output to inference_identifier
            inference_output = self.inference_identifier(segmenter_output)
            inter_map_inference_output = {'aif':{'nodes':[],
                                                 'edges':[],
                                                 'locutions':[]}}
            if inter_map_segments:
                inter_map_inference_output = self.inference_identifier(inter_map_segments)

            # Process inference output based on inference_identifier_QUEUE
            if self.inference_identifier_QUEUE:
                self.previous_map = self.inference_identifier_QUEUE.popleft()
                current_map = inference_output
                merged_map = self.merge_map(current_map, self.previous_map,inter_map_inference_output)
                self.inference_identifier_QUEUE.append(merged_map)
            else:
                self.inference_identifier_QUEUE.append(inference_output)

            # Append segmenter output to segmenter_QUEUE
            self.segmenter_QUEUE.append(segmenter_output)

        # Continue processing until segmenter_QUEUE is empty
        while self.segmenter_QUEUE:
            segmenter_output = self.segmenter_QUEUE.popleft()
            inference_output = self.inference_identifier(segmenter_output)
            self.previous_map = self.inference_identifier_QUEUE.popleft()
            current_map = inference_output
            merged_map = self.merge_map(current_map, self.previous_map)
            self.inference_identifier_QUEUE.append(merged_map)

        return self.inference_identifier_QUEUE


# Initialize TextProcessor
text_processor = TextProcessor()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_obj = request.files.get('file')
        f_name = file_obj.filename
        file_obj.save(f_name)
        file = open(f_name,'r')
        text_processor.process_file(f_name)

    return render_template('index.html')

# WebSocket event handler to send merged output to the client
@socketio.on('request_merged_output')
def handle_request_merged_output():
    while text_processor.inference_identifier_QUEUE:
        # Get merged output
        merged_output = text_processor.inference_identifier_QUEUE.popleft()
        # Emit merged output to the client
        emit('merged_output', {'output': merged_output})






if __name__ == '__main__':
    socketio.run(app, debug=True)
