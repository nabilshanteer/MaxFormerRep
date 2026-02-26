## Extensions

To extend the original work, we could investigate the role of the LIF decay factor β (membrane time constant) on frequency selectivity and performance. The paper theoretically shows that β controls the pole of the low-pass transfer function, but does not empirically test this. We will train Max-Former on CIFAR-100 with multiple β values (e.g., β ∈ {0.1,0.25,0.5,0.75}) while keeping all other parameters fixed. This experiment will test whether shorter time constants improve high-frequency propagation and accuracy, as predicted by the theory. Each additional β configuration will require one full training run (if set up similarly to the experiments in the paper), which will take a day each.


Alternatively, while the authors attribute performance gains to high-frequency restoration via MaxPooling, it is unclear whether this was due to MaxPool specifically or generally to high-pass components. Thus, another possible extension could involve investigating the performance of different fixed and learning high-pass filters (such as the Laplacian Filter) and comparing performance on the CIFAR-100 dataset.
