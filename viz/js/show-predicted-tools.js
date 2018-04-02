$(document).ready(function() {
    let toolsData = null,
        //toolsSeqTemplate = "",
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

    function makeToolSequence( toolSeq ) {
        let toolsSplit = toolSeq.split( "," ),
            template = "",
            lenTools = toolsSplit.length;
        for( let counter = 0; counter < lenTools; counter++ ) {
            let currTool = toolsSplit[ counter ];
            if( lenTools === 1 ) {
                template = currTool;
            }
            else {
                if ( counter === lenTools - 1 ) {
                    template += currTool;
                }
                else {
                    template += currTool + "<span class='arrow-tool'> &rarr; </span>";
                }
                
            }
        }
        return template;
    }

    function refreshPrediction( selectedTool ) {
        let $elWf = $( ".workflow-so-far" );
        $elWf.empty();
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
        template = makeToolSequence( toolsSeq );
        getPredictedTools( toolsSeq );
        $elWf.append( template );
        $( ".tool-ids" ).prop( 'selectedIndex', 0 );
        registerRemoveTool();
    }

    function registerRemoveTool() {
        $( ".remove-last-tool" ).off( "click" ).on( "click", function( e ) {
            refreshPrediction( "" );
        });
    }

    function sortDictionary( dictionary ) {
        let toolsList = [];
        for( var key in dictionary ) {
            toolsList.push( dictionary[ key ] )
        }
        return toolsList.sort();
    }

    function getPredictedTools( toolSeq ) {
        let predictUrl =  "http://" + host + "/?predict_tool=true&tool_seq=" + toolSeq,
            $elLoading = $( ".loading-image" ),
            $elActualPredictedTools = $( ".predicted-actual-tools" ),
            $elAllPaths = $( ".all-included-paths" );
        if( toolSeq === "" )
            return;
        $elLoading.show();
        $elActualPredictedTools.empty();
        $elAllPaths.empty();
        $.getJSON( predictUrl, function( data ) {
            console.log(data);
            let predictedNodes = data[ "predicted_nodes" ],
                allInputPaths = data[ "all_input_paths" ],
                predictedProb = data[ "predicted_prob" ],
                correctPredictedNodes = data[ "actual_predicted_nodes" ],
                actualNextNodes = data[ "actual_labels" ],
                actualLabelDist = data[ "actual_labels_distribution" ],
                presentCounter = 0,
                predictionRate = 0.0,
                toolsTemplate = "",
                correctToolsTemplate = "",
                pathsTemplate = "",
                topK = actualLabelDist.length;
            if( Object.keys( predictedNodes ).length > 0 ) {
                predictedNodeList = predictedNodes.split( "," );
                toolsTemplate = "<table class='table table-bordered table-striped thead-dark'>";
                toolsTemplate += "<thead><th>SNo.</th><th>Predicted next tools</th><th> Actual next tools</th></thead>";
                toolsTemplate += "<tbody>";
                for( let counter = 0; counter < topK; counter++ ) {
                    let nodeName = predictedNodeList[ counter ],
                        isTrue = correctPredictedNodes[ nodeName ],
                        actualProb = ( actualLabelDist[ counter ][ 1 ] * 100 ).toFixed( 2 ),
                        prob = ( predictedProb[ counter ] * 100 ).toFixed( 2 );
                    toolsTemplate += "<tr>";
                    toolsTemplate += "<td>" + ( counter + 1 ) + "</td>";
                    if ( isTrue || isTrue === 'true' ) {
                        toolsTemplate += "<td class='present-node'>" + nodeName + "(" + prob + "%)" + "</td>";
                        presentCounter += 1;
                    }
                    else {
                        toolsTemplate += "<td class='absent-node'>" + nodeName + "(" + prob + "%)" + "</td>";
                    }
                    toolsTemplate += "<td class='present-node'>" + actualLabelDist[ counter ][ 0 ] + "(" + actualProb + "%)" + "</td>";
                    toolsTemplate += "</tr>";
                }
                toolsTemplate += "</tbody></table>";
                predictionRate = ( presentCounter / actualNextNodes.length ) * 100;
                toolsTemplate += "<div>Prediction rate: <b>" + predictionRate.toFixed( 2 ) + "%</b></div>";
            }
            else {
                toolsTemplate = "No predicted tools";
            }

            if( Object.keys( allInputPaths ).length > 0 ) {
                pathsTemplate = "<ol>";
                for( var index in allInputPaths ) {
                    let toolsPath = allInputPaths[ index ],
                        toolItems = toolsPath.split( "," );
                    if( toolItems.length > 1 ) {
                        toolsPath = toolsPath.split( "," ).join( " &rarr; " );
                        pathsTemplate += "<li>" + toolsPath + "</li>";
                    }
                }
                pathsTemplate += "</ol>";
            }
            else {
                pathsTemplate = "No paths involving this sequence in the complete data"
            }
            $elLoading.hide();
            $elActualPredictedTools.append( toolsTemplate );
            $elAllPaths.append( pathsTemplate );
        });
    }
    function clearDataEvent() {
        $( ".clear-data" ).on( "click", function( e ) {
            document.location.reload();
        });
    }
    clearDataEvent();
});

