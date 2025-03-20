process DEEPTMHMM {
    tag "$meta.id"
    label 'process_single'

    // Use your Docker Hub image with predict.py
    container 'docker://sandyjmacdonald/deeptmhmm:0.9.4'

    publishDir "$outdir", mode: "copy"

    input:
    tuple val(meta), path(fasta)
    val outdir

    output:
    tuple val(meta), path("${meta.id}/TMRs.gff3")                 , emit: gff3
    tuple val(meta), path("${meta.id}/predicted_topologies.3line"), emit: line3
    tuple val(meta), path("${meta.id}/deeptmhmm_results.md")      , emit: md
    tuple val(meta), path("${meta.id}/*_probs.csv")               , optional: true, emit: csv
    tuple val(meta), path("${meta.id}/plot.png")                  , optional: true, emit: png
    path "versions.yml"                                               , emit: versions

    script:
    def args = task.ext.args ?: ''
    def fasta_name = fasta.name
    // Define the output directory using meta.id
    def output_dir = "${meta.id}"
    
    """
    # Run predict.py using the provided fasta file and output results to the directory named after meta.id
    deeptmhmm-predict --fasta ${fasta_name} --output-dir ${output_dir} ${args}

    # Write a simple versions file
    echo "deeptmhmm-predict version: 1.0.0" > versions.yml
    """
}