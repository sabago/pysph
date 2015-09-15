# Introduction #

Since computational numbers are finite, the code has some limits for various things which are done.
Most of the developers work on 64 bit linux machines and hence these are valid for them, however they may also be valid in some other architectures.


# Details #

Particle positions and all other properties : double precision

Cell indices : stored as integers (IntPoint) but absolute value can only be 20 bit maximum (1048576) in each dimension due to hash function being only a long (64 bit). Higher indices may cause inexplicable errors as no checking is performed

Max number of particles : maximum size of a long