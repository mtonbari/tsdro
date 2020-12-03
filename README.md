 # Two-stage Distributionally Robust Optimization in Disaster Management
 
 Implementation of various algorithms to solve a distributionally robust two-stage disaster management model. The first stage is a facility location problem where decisions are which facilities to open and where, and how many resources to allocate at each open facility. The second stage is a routing problem of the resources to affected areas after observing a natural disaster. 
 
 ## Examples
 Two example scripts can be found in the src folder:
 
    *run_tsdro.py: solves the two-stage distributionally robust model using a Wasserstein ambiguity set.
    *run_saa.py: solves the two-stage model using Sample Average Approximation.
  
## References
Carmen G. Rawls and Mark A. Turnquist, Pre-positioning of emergency supplies for disaster response, 2010.
