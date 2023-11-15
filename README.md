 # Two-stage Disaster Management Optimization
 
 Implementation of various algorithms to solve a two-stage disaster management model. The first stage is a facility location problem where decisions are which facilities to open and where, and how many resources to allocate at each open facility. The second stage is a routing problem of the resources to affected areas after observing a natural disaster.

Implementations include a classical Sample Average Approximation (SAA), and a column-and-constraint generation algorithm to solve the two-stage distributionally robust variant under the Wasserstein set.

 
 ## Examples
 Two example scripts can be found in the src folder:
 
    *run_tsdro.py: solves the two-stage distributionally robust model using a Wasserstein ambiguity set.
    *run_saa.py: solves the two-stage model using Sample Average Approximation.
  
## References
Carmen G. Rawls and Mark A. Turnquist, Pre-positioning of emergency supplies for disaster response, 2010.
