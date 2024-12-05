from flask import Flask, request, jsonify
from flask_cors import CORS 
from convert2pytorch import GraphData


app = Flask(__name__)
CORS(app)  # Enable CORS
PORT = 5000

@app.route('/api/saveGraph', methods=['POST'])
def save_graph():
    try:
        graph_data = request.json  # Get JSON data from request
        print("Graph Data Received:", graph_data)
        print("\n")
        # Process the graph data (e.g., convert to PyTorch)
        try:
            graph = GraphData(graph_data)
            network_result = graph.build_network()
            print("Network Result:", network_result)
            return jsonify({
                'status': 'success',
                'code': network_result['code'],  # Send the generated code
                'message': 'Model created successfully'
            }), 200
        except Exception as e:
            print("Error in GraphData processing:", str(e))
            return jsonify({'status': 'error', 'message': f"GraphData processing error: {str(e)}"}), 500
        
    except Exception as e:
        print("General Error:", str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=PORT)