cpdef compute_star(int dest_pid, int source_pid,
                   double gamma,
                   double rhol, double rhor,
                   double pl, double pr,
                   double ul, double ur)

cdef prefun(double p, double dk, double pk, double ck, double gamma)

cpdef sample(double pm, double um, double s,
             double rhol, double rhor,
             double pl, double pr,
             double ul, double ur,
             double gamma)
