/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "gamma_replayer.h"

int
main(int argc, char **argv)
{
   uint32_t num_test_runs = 5;
   uint32_t permute_seed = 0;
   char *out = NULL;
   char *in = NULL;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--help")) {
         in = NULL;
         break;
      }
      if (!strcmp(argv[i], "--test-runs")) {
         if (i == argc - 1)
            goto error_args;
         num_test_runs = atoi(argv[++i]);
         continue;
      }
      if (!strcmp(argv[i], "--permute-seed")) {
         if (i == argc - 1)
            goto error_args;
         permute_seed = atoi(argv[++i]);
         continue;
      }
      if (!strcmp(argv[i], "-o")) {
         if (i >= argc - 1)
            goto error_args;
         out = argv[++i];
         continue;
      }
      if (!strcmp(argv[i], "-i"))
         i++;
      in = argv[i];
   }

   if (!num_test_runs || !in || !out) {
error_args:
      fprintf(stderr,
              "Usage: %s [--test-runs N] [--permute-seed N] <-o csvfile> infile\n"
              "test-runs specifies the number of runs to perform. Default: 5\n"
              "permute-seed specifies an optional random seed for deterministic permutation of runs. Default: 0 "
              "(disable permutation)\n"
              "csvfile is a path to a .csv file to dump benchmark results to.\n"
              "Arguments can occur in any order. Input files may also be explicitly prefixed with -i.\n",
              argv[0]);
      return 1;
   }

   gamma_replayer replayer = gamma_replayer();
   if (gamma_replayer_init(&replayer)) {
      fprintf(stderr, "gamma: Failed to initialize replayer.\n");
      return 1;
   }

   replayer.export_csv = out;

   FILE *f = fopen(in, "rb");
   if (!f) {
      fprintf(stderr, "gamma: Failed to open file %s.\n", in);
      return 1;
   }
   FILE *outfile = fopen(out, "w");
   if (!outfile) {
      fprintf(stderr, "gamma: Failed to open file %s.\n", out);
      return 1;
   }

   replayer.load(f);
   fclose(f);

   if (replayer.builds.empty()) {
      fprintf(stderr, "gamma: No acceleration structure build info found in %s.\n", in);
      return 1;
   }

   gamma_replayer_run(&replayer, permute_seed, num_test_runs);

   replayer.print_results(in, outfile);
   fclose(outfile);
}
