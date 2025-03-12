process DEEPTMHMM {
    tag "$meta.id"
    label 'process_single'

    // Use your Docker Hub image with predict.py
    container 'docker://sandyjmacdonald/deeptmhmm:0.1.0'

    publishDir "$outdir/${meta.id}", mode: "copy"

    input:
    tuple val(meta), path(fasta)
    val outdir

    output:
    tuple val(meta), path("results/TMRs.gff3")                 , emit: gff3
    tuple val(meta), path("results/predicted_topologies.3line"), emit: line3
    tuple val(meta), path("results/deeptmhmm_results.md")      , emit: md
    tuple val(meta), path("results/*_probs.csv")               , optional: true, emit: csv
    tuple val(meta), path("results/plot.png")                  , optional: true, emit: png
    path "versions.yml"                                               , emit: versions

    script:
    def args = task.ext.args ?: ''
    // Use the fasta file name directly
    def fasta_name = fasta.name

    """
    # Run predict.py using the provided fasta file and output results to the "results" directory
    cd /app
    python3 predict.py --fasta ${fasta_name} --output-dir results ${args}

    # Write a simple versions file
    echo "predict.py version: 1.0.0" > versions.yml
    """
}