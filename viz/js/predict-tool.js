$(document).ready(function() {
    var workflowsData = null,
        path = "https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/master/viz/data/workflows.json";
    if ( path === "" ) {
        console.error( "Error in loading JSON file" );
        return;
    }

    $.getJSON( path, function( data ) {
        let workflowTemplate = "";
        workflowsData = data;
        for( let id in workflowsData ) {
            if( id !== undefined ) {
                workflowTemplate += "<option value='" + id + "'>" + id + "</options>";
            }
        } // end of for loop
        $( ".workflow-ids" ).append( workflowTemplate );
    });

    // Fire on change of a tool to show similar tools and plot cost vs iteration
    $( ".workflow-ids" ).on( 'change', function( e ) {
        e.preventDefault();
        let selectedWorkflowId = e.target.value,
            $elWorkflows = $( ".workflow-details" ),
            workflowTableTemplate = "",
            workflowDetails = workflowsData[ selectedWorkflowId ];
        $elWorkflows.empty();
        workflowTableTemplate = createHTML( workflowDetails );
        $elWorkflows.append( workflowTableTemplate );
        buildCytoscapeGraphData( workflowDetails );
    });

    var createHTML = function( workflowDetails ) {
        let template = "",
            workflowSteps = workflowDetails[ "original_steps" ];
        template = "<div class='table-header-text'>Workflow id: " + workflowDetails[ "id" ] + "</div>";
        template += "<div class='table-header-text'>Workflow name: " + workflowDetails[ "name" ] + "</div>";
        template += "<div class='table-responsive'><table class='table table-bordered table-striped thead-dark table-steps'><thead>";
        template += "<th>Step no. </th>";
        template += "<th>Name</th>";
        template += "<th>Type</th>";
        template += "<th>Tool id</th>";
        template += "<th> Input connections </th>";
        template += "</thead><tbody>";

        for( let counter = 0, len = workflowSteps.length; counter < len; counter++ ) {
            let step = workflowSteps[ counter ],
                inputConnections = step[ "input_connections" ],
                stepInputs = [];
            for( var ic in inputConnections ) {
                stepInputs.push( inputConnections[ ic ].id );
            }
            template += "<tr>";
            template += "<td>" + step.id + "</td>";
            template += "<td>" + step.name + "</td>";
            template += "<td>" + step.type + "</td>";
            template += "<td>" + step.tool_id + "</td>";
            template += "<td>" + stepInputs.join( "," ) + "</td>";
            template += "</tr>";
        }
        template += "</tbody></table></div>";
        return template;
    };

    /** Build data for generating cytoscape graphs */
    var buildCytoscapeGraphData = function( workflow ) {
        let $elStepsGraph = document.getElementById( 'workflow-graph' ),
            stepNodes = [],
            stepEdges = [],
            workflowSteps = [];
        workflowSteps = workflow[ "original_steps" ];
        _.each( workflowSteps, function( step ) {
            let targetNode = step[ "id" ],
                inputConnections = {},
                stepInputs = [],
                name = "";
            if( !step.tool_id ) {
                name = step[ "name" ];
            }
            else if( step.tool_id.split( "/" ).length > 1 ) {
                name = step.tool_id.split( "/" );
                name = name[ name.length - 2 ];
            }
            else {
                name = step.tool_id
            }
            inputConnections = step[ "input_connections" ];
            stepNodes.push( { data: { id: targetNode, weight: 0.25, name: step[ "id" ] + ", " + name } } );
            for( var ic in inputConnections ) {
                stepInputs.push( inputConnections[ ic ].id );
            }
            if( stepInputs.length > 0 ) {
                _.each( stepInputs, function( sourceNode ) {
                    stepEdges.push( { data: { source: sourceNode, target: targetNode, weight: 0.25 } } );
                });
            }
        });
        
        // make call to cytoscape to generate graphs
        var cytoscapePromise = new Promise( function( resolve, reject ) {
            makeCytoGraph( { elem: $elStepsGraph, nodes: stepNodes, edges: stepEdges } );
        });
    };

    /** Create cytoscape graph */
    var makeCytoGraph = function( data ) {
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

        return graph;
    };
});

