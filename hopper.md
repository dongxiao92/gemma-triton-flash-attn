# Port attention,py lernel for gemma4 on H100

## Goal
The goal is to understand the current kernel implementation in this repo, and then port it to support Hopper(h100). The ported kernel should match the same functionality and pass functionality test for gemma4–e2b–it model.
One known issue is tha the global attention part is not supported, which seems to be related to a headdim=512. There might be other issues also.
After a functionality version, do optimizations and achive >70 SOL.

## Requirement
1. use computelab-sc-01 to get H100 nodes for testing.
2. use gemma4–e2b–it + transformer as the functionality baseline to align with.
3. Use a local dev branch to track the work.
