#!/usr/bin/env python3
"""module"""


class Normal:
    """ represents a normal distribution """

    p = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            s = 0
            for x in range(len(data)):
                s += (data[x] - self.mean) ** 2
            s = s / len(data)
            self.stddev = float(s ** (1/2))

    def z_score(self, x):
        """Calculates the z-score of a given x-value """
        z_c = (x - self.mean) / self.stddev
        return(z_c)

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        x_v = self.mean + (self.stddev * z)
        return(x_v)

    def pdf(self, x):
        "Calculates the value of the PDF for a given x-value"
        pdf1 = 1 / (self.stddev*(2*self.p) ** (1/2))
        pdf_a = pdf1 * (self.e ** ((-1/2)*((x-self.mean) / self.stddev) ** 2))
        return(pdf_a)

    def cdf(self, x):
        "Calculates the value of the CDF for a given x-value"

        c = (x - self.mean) / (self.stddev*(2**(1/2)))
        erf = (2/(self.p**(1/2))*(c - c**3/3 + c**5/10 - c**7/42 + c**9/216))
        cdf = (1/2) * (1 + erf)
        return(cdf)
