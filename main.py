from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from collections import deque

app = Flask(__name__)
socketio = SocketIO(app)

class TextProcessor:
    def __init__(self):
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

    def turninator(self, sentences):
        #print("Turninator Input:", sentences)
        output = [sentence.upper() for sentence in sentences]
        #print("Turninator Output:", output)
        return output

    def segmenter(self, sentences):
        #print("Segmenter Input:", sentences)
        output = [" segment " + sentence for sentence in sentences]
        #print("Segmenter Output:", output)
        return output

    def inference_identifier(self, segmented_sentences):
        #print("Inference Identifier Input:", segmented_sentences)
        output = [' Default Inference' + segment for segment in segmented_sentences]
        #print("Inference Identifier Output:", output)
        return output

    def merge_map(self, current_map, previous_map):
        if current_map is None:
            return previous_map
        elif previous_map is None:
            return current_map
        else:
            output = [c + p for c, p in zip(current_map, previous_map)]
            self.output = output  # Store the merged output
            return output


    def get_topics(self, segmented_sentences):
        segment_topic_dict = {}
        for segment in segmented_sentences:
            if segment in segment_topic_dict:
                segment_topic_dict[segment].append(("nodeID", "topic"))
            else:
                segment_topic_dict[segment] = [("nodeID", "topic")]
        return segment_topic_dict

    def create_linking_map(self, segment_topic_dict):
        if not segment_topic_dict:
            return {}, {}
        xAIF = {}
        return xAIF, segment_topic_dict

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
            if inter_map_segments:
                inter_map_inference_output = self.inference_identifier(segmenter_output)

            # Process inference output based on inference_identifier_QUEUE
            if self.inference_identifier_QUEUE:
                self.previous_map = self.inference_identifier_QUEUE.popleft()
                current_map = inference_output
                merged_map = self.merge_map(current_map, self.previous_map)
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

# Route to render the HTML page
@app.route('/')
def index():
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
