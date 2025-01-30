#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { DEEPTMHMM } from './modules/nf-core/deeptmhmm'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow NFCORE_TRANSMEMBRANE {

    take:
    input_samplesheet // Define an input channel for the workflow

    main:

    // Ensure input is provided
    if (!params.input) {
        error "ERROR: No input samplesheet provided! Use --input <samplesheet.csv>"
    }

    // Read samplesheet and create proper channel
    ch_fasta = Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row -> 
            tuple([ id: row.sequence ], file(row.fasta)) // Correctly format the tuple
        }
        .set { fasta_channel }

    // Call DEEPTMHMM module with formatted channel
    DEEPTMHMM(fasta_channel)
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:

    // Create the input channel for the workflow
    ch_input = Channel.value(params.input)

    // Pass input channel to the workflow
    NFCORE_TRANSMEMBRANE(ch_input)
}