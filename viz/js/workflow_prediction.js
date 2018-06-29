$(document).ready(function() {
    let toolsData = null,
        toolsSeq = "",
        host = document.location.host,
        toolsUrl = "http://" + host + "/tools_list=true";
    if ( toolsUrl === "" ) {
        console.error( "Error in loading text file" );
        return;
    }
    $.getJSON( toolsUrl, function( data ) {
        let toolsListOptions = "";
        toolsData = sortDictionary( data );
        for( let counter = 0; counter < toolsData.length; counter++ ) {
            toolsListOptions += "<option value='" + counter + "'>" + toolsData[ counter ] + "</options>";
        }
        $( ".tool-ids" ).append( toolsListOptions );
    });

    // Fire on change of a workflow to show all the steps for the workflow and its directed graph
    $( ".tool-ids" ).on( 'change', function( e ) {
        refreshPrediction( e.target.value );
    });

    function refreshPrediction( selectedTool ) {
        if( selectedTool !== "" ) { 
            if( toolsSeq === "" ) {
                toolsSeq = toolsData[ selectedTool ];
            }
            else {
                toolsSeq += "," + toolsData[ selectedTool ];
            }
        }
        else {
            toolsSeqSplit = toolsSeq.split( "," );
            toolsSeq = toolsSeqSplit.slice( 0, toolsSeqSplit.length - 1 );
            toolsSeq = toolsSeq.join( "," );
        }
        getPredictedTools( toolsSeq );
        $( ".tool-ids" ).prop( 'selectedIndex', 0 );
    }

    function sortDictionary( dictionary ) {
        let toolsList = [];
        for( var key in dictionary ) {
            toolsList.push( dictionary[ key ] )
        }
        return toolsList.sort();
    }

    /** Create cytoscape graph */
    function makeCytoGraph( data ) {
        let graph = cytoscape({
            container: data.elem,
            elements: {
                nodes: data.nodes,
                edges: data.edges
            },
            layout: {
                name: 'cose'
            },
            style: [
                {
                    selector: 'node',
                    style: {
                        'content': 'data(name)',
                        'text-opacity': 1,
                        'text-valign': 'center',
                        'text-halign': 'right',
                        'background-color': '#337ab7',
                        'font-size': '9pt',
                        'font-family': '"Lucida Grande", verdana, arial, helvetica, sans-serif'
                    }
                },

                {
                    selector: 'edge',
                    style: {
                        'line-color': '#9dbaea',
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'font-size': '9pt',
                        'font-family': '"Lucida Grande", verdana, arial, helvetica, sans-serif'
                    }
                }
            ]
        });

        // when a node is tapped, make a search with the node's text
        graph.on( 'tap', 'node', function( ev ) {
            let nodeId = this.id();
            for( let counter = 0; counter < toolsData.length; counter++ ) {
                if ( toolsData[ counter ] === nodeId ) {
                    refreshPrediction( counter );
                    break;
                }
            }
        });

         // color the source node differently
        _.each( data.predictedNodes, function( item ) {
            let $graphElem = graph.$( "#" + item );
            $graphElem.style( "backgroundColor","green" );
        });

        return graph;
    }

    function getPredictedTools( toolSeq ) {
        let predictUrl =  "http://" + host + "/?predict_tool=true&tool_seq=" + toolSeq,
            $elWorkflowGraph = document.getElementById( 'workflow-graph' );
        $( "#workflow-graph" ).empty();
        $( "#workflow-graph" ).html( "<span class='load-graph'>Loading graph...</span>" );
        $.getJSON( predictUrl, function( data ) {
            let predictedNodes = data[ "predicted_nodes" ],
                predictedProb = data[ "predicted_prob" ],
                topK = 5,
                cytoNodes = [],
                cytoEdges = [],
                lastToolSplit = toolSeq.split( "," ),
                lastTool = lastToolSplit[ lastToolSplit.length - 1 ];
            if( Object.keys( predictedNodes ).length > 0 ) {
                predictedNodeList = predictedNodes.split( "," );
                cytoNodes.push( { data: { id: lastTool, weight: 0.1, name: lastTool } } );
                for( let counter = 0; counter < topK; counter++ ) {
                    let nodeName = predictedNodeList[ counter ],
                        prob = ( predictedProb[ counter ] * 100 ).toFixed( 2 );
                    cytoNodes.push( { data: { id: nodeName, weight: 0.1, name: nodeName + " (" + prob + "%)" } } );
                    cytoEdges.push( { data: { source: lastTool, target: nodeName, weight: 0.1 } } );
                }
                for( let counter = 0; counter < lastToolSplit.length - 1; counter++ ) {
                    cytoNodes.push( { data: { id: lastToolSplit[ counter ], weight: 0.1, name: lastToolSplit[ counter ] } } );
                    cytoEdges.push( { data: { source: lastToolSplit[ counter ], target: lastToolSplit[ counter + 1 ], weight: 0.1 } } );
                }
            }
            // make call to cytoscape to generate graphs
            var cytoscapePromise = new Promise( function( resolve, reject ) {
                $( "#workflow-graph" ).empty();
                makeCytoGraph( { elem: $elWorkflowGraph, nodes: cytoNodes, edges: cytoEdges, predictedNodes: predictedNodeList } );
                //$( "#workflow-graph" ).removeClass( "fade-out" );
            });
        });
    }
    function clearDataEvent() {
        $( ".clear-data" ).on( "click", function( e ) {
            document.location.reload();
        });
    }
    clearDataEvent();
});
