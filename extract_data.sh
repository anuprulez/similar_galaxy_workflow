#!/bin/bash
# Create dir if ont existing
mkdir -p data/

# Run the queries
psql > data/tool-popularity.tsv <<-EOF
	COPY (
		SELECT
			tool_id,
			date_trunc('day', create_time AT TIME ZONE 'UTC')::date as day,
			count(*)
		FROM job
		WHERE create_time > (now() AT TIME ZONE 'UTC' - '24 months'::interval)
		GROUP BY tool_id, day
		ORDER BY day desc, count desc
	) to STDOUT with CSV DELIMITER E'\t'
EOF

psql > data/wf-connections.tsv <<-EOF
	COPY (
		SELECT
			ws_in.workflow_id as wf_id,
			date_trunc('minute', workflow.update_time AT TIME ZONE 'UTC') as wf_updated,
			wfc.input_step_id as in_id,
			ws_in.tool_id as in_tool,
			ws_in.tool_version as in_tool_v,
			wfc.output_step_id as out_id,
			ws_out.tool_id as out_tool,
			ws_out.tool_version as out_tool_v
		FROM
			workflow_step_connection wfc
		JOIN workflow_step ws_in  ON ws_in.id = wfc.input_step_id
		JOIN workflow_step ws_out ON ws_out.id = wfc.output_step_id
		JOIN workflow on ws_in.workflow_id = workflow.id
		WHERE
			workflow.id in (
				SELECT
				 workflow.id
				FROM
				 stored_workflow
				LEFT JOIN
				 workflow on stored_workflow.latest_workflow_id = workflow.id
			)
	) to STDOUT with CSV DELIMITER E'\t'
EOF
