#include <fstream>
#include <libs/json.hpp>
#include <MTensor/graph.hpp>

void export_as_html(std::ofstream &out, json &data){

        out << R"(<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Gradient Graph</title>
  <style>html, body, #cy { width: 100%; height: 100%; margin: 0; padding: 0; }</style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.32.0/cytoscape.min.js" ></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js" integrity="sha512-psLUZfcgPmi012lcpVHkWoOqyztollwCGu4w/mXijFMK/YcdUdP06voJNVOJ7f/dUIlO2tGlDLuypRyXX2lcvQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
  <style>

#info-panel * {
    box-sizing: border-box;
}

#info-panel ,  #general-info-panel{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #fff;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    color: #333;
    padding: 20px 25px;

    /* Positioning & Hiding */
    position: absolute;
    top: 20px;
    right: 20px;
    width: 320px;
    z-index: 10;
    
    /* Initially hidden */
    display: none;
}

#general-info-panel{
  left:  20px;
  right:  auto;
  display: block;
}

#info-panel h3 , #general-info-panel h3 {
    margin-top: 0;
    margin-bottom: 20px;
    font-size: 18px;
    color: #000;
    text-align: center;
}

#info-panel p , #general-info-panel p{
    margin: 0 0 10px;
    line-height: 1.6;
    font-size: 14px;
}

#info-panel strong , #general-info-panel strong{
    color: #555;
    font-weight: 600;
}

#close-button {
    position: absolute;
    top: 15px;
    right: 15px;
    background: #f1f1f1;
    border: none;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    color: #555;
    font-size: 14px;
    line-height: 28px;
    text-align: center;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
}

#close-button:hover {
    background-color: #e6e6e6;
    color: #000;
}
  </style>
</head>
<body>
<div id="cy" style="width:95%; height : 95vh; margin : auto;"></div>
<div id="info-panel">
  <button id="close-button">X</button>
  <div id="info-content"></div>
</div>
<div id="general-info-panel">
  <h3> general information </h3>
  <div id="general-info-content"></div>
</div>
<script>
const general_info = )";
      out << data["general"].dump(2);
      out << R"(;     
const data = 
)";

        out << data["graph_data"].dump(2);

        out << R"(;

const cyInstance = cytoscape({
  container: document.getElementById('cy'),
  elements: data,
      style: [
        {
          selector: 'node.op',
          style: {
            'shape': 'roundrectangle',
            'background-color': 'red',
            'width': '120px',
            'height': '40px',
            'label': 'data(label)',
            'color': '#fff',
            'font-size': '12px',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-radius': '5px',
          }
        },
        {
          selector: 'node.tensor',
          style: {
            'shape': 'roundrectangle',
            'background-color': '#3498db',
            'width': '200px',              
            'height': 'auto',              
            'label': 'data(label)',
            'color': 'white',
            'font-size': '13px',
            'text-wrap': 'wrap',
            'text-max-width': '180px',     
            'text-valign': 'center',
            'text-halign': 'center',
            'border-radius': '10px',
            'padding': '10px',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#ccc',
            'target-arrow-color': '#ccc',
            'target-arrow-shape': 'triangle'
          }
        }
      ],
      layout: {
        name: 'dagre', // Use dagre layout for DAGs
        rankDir: 'TB',
        nodeSep: 40,
        edgeSep: 10,
        rankSep: 50
      }
    });
// Get the panel elements
const infoPanel = document.getElementById('info-panel');
const infoContent = document.getElementById('info-content');
const generalContent = document.getElementById('general-info-content');
const closeButton = document.getElementById('close-button');

// --- Event Listeners ---

// Show panel on tensor node click
cyInstance.on('tap', 'node.tensor', function(evt){
  const node = evt.target;
  const nodeData = node.data();
  details = "";
  grad_details = "";

  if (nodeData.details){
      for (const el in nodeData.details){
        details += `<p><strong>${el} : </strong> ${nodeData.details[el]}</p>`
      } 
  }

  if (nodeData.grad_details){
      grad_details += 
      `<p><strong>ID:</strong> ${nodeData.id}</p>
      <p><strong>Label:</strong> ${nodeData.label}</p>`
      for (const el in nodeData.grad_details.details){
        grad_details += `<p><strong>${el} : </strong> ${nodeData.grad_details.details[el]}</p>`
      } 
  }

  infoContent.innerHTML = `
    <p><strong>ID:</strong> ${nodeData.id}</p>
    <p><strong>Label:</strong> ${nodeData.label}</p>
    <p><strong style="color:blue;">Details:</strong> ${nodeData.details ? details : 'No additional details available.'}</p>
    <hr/>
    <p><strong style="color:green;">grad Details:</strong> ${nodeData.grad_details ? grad_details : 'No grad details.'}</p>
  `;
  
  // Instantly show the panel
  infoPanel.style.display = 'block';
});

// Hide panel when the close button is clicked
closeButton.addEventListener('click', function() {
  infoPanel.style.display = 'none';
});

// Hide the panel if you click on the graph background
cyInstance.on('tap', function(event){
  // Check if the tap was on the core canvas and not a node or edge
  if (event.target === cyInstance) {
    infoPanel.style.display = 'none';
  }
});



for (const el in general_info){
  generalContent.innerHTML += `<p><strong>${el} : </strong> ${general_info[el]}</p>`
} 

  </script>
</body>
</html>)";
}

