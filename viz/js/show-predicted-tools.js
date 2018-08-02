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
                    template += currTool + "<span>,</span>"; //&rarr; //class='arrow-tool'
                }
            }
        }
        return template;
    }

    function refreshPrediction( selectedTool ) {
        console.log(selectedTool);
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

    function clearDataEvent() {
        $( ".clear-data" ).on( "click", function( e ) {
            document.location.reload();
        });
    }
    clearDataEvent();
});

