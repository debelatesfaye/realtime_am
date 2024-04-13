from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics

from default_turninator.src.data import Data
from default_turninator.src.turninator import Turninator 
from default_turninator.src.util import handle_errors
import logging
#logging configuration
logging.basicConfig(datefmt='%H:%M:%S',
                    level=logging.DEBUG)


app = Flask(__name__)
metrics = PrometheusMetrics(app)
	
@app.route('/turninator-01', methods = ['GET', 'POST'])
@metrics.summary('requests_by_status', 'Request latencies by status',
                 labels={'status': lambda r: r.status_code})
@metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
                   labels={'status': lambda r: r.status_code, 'path': lambda: request.path})
@handle_errors  
def turninator_defult():
	if request.method == 'POST':
		file_obj = request.files.get('file')	
		data = Data(file_obj)	
		turninator = Turninator()
		result=turninator.turninator_default(file_obj)
		return result	

	if request.method == 'GET':
		info = """Turninator is an AMF component that parses arguments into dialog turns.  
		This is the default implementation of a turninator that parses arguments using simple regular expressions. 
		It expects dialogical texts to be specified in the form of 'speaker:text' formats. 
		It takes input in both text and xIAF formats (where the input is specified in the text field of xIAF) to return xIAF as an output. 
		Please note that the argument type (dialog vs monolog) can be specified in xIAF if the input is in xIAF format; otherwise, it is considered a monological argument. 
		The component can be used as a starting point for creating an argument mining pipeline that takes monological or dialogical text specified as regular text or xIAF."""
		return info
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5006"), debug=False)	  
