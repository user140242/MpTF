/*
MIT License

Copyright (c) 2024 user140242

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

///    This is a implementation of the Mersenne numbers trial factoring with GMP and OpenMP
///    MpTF v. 1.0.3

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <stdint.h>
#include <gmp.h>
#include <omp.h>
#include <chrono>

const unsigned long long k_segment_size = 524288;
//vector segment size  is (k_segment_size * nR / 8)
//It is possible to vary n_PB_v to change the modulus and the maximum number of classes useful nR   
//n_PB_v == 1 useful num classes nR = 16, n_PB_v == 2 useful num classes nR = 96, n_PB_v == 3 useful num classes  nR = 960
const int n_PB_v = 1;

const uint8_t del_bit[8] =
{
  0x7f, 0xbf, 0xdf, 0xef, 0xf7, 0xfb, 0xfd, 0xfe
};

long long modinv(long long a, long long  modulus)
{
    // insert a<modulus and modulus with gcd(a,modulus) = 1 return b with a * b = 1 % modulus
    long long x_p = 1;
    if (a > 1 && modulus > 1)
    {
        x_p = 0;
        long long x_m = 1;
        long long x_t;
        long long r_t = a;
        long long r_m = modulus;
        long long r_p = r_t - r_m * (r_t / r_m);
        
        while (r_p > 0)
        {
            x_t = x_p;
            x_p = x_m - (r_t / r_m) * x_t;
            x_m = x_t;
            r_t = r_m;
            r_m = r_p;
            r_p = r_t - r_m * (r_t / r_m);
        }
    }
    return x_p;
}

unsigned long long mulmod(unsigned long long X, unsigned long long Y, unsigned long long D)
{
    // return (X * Y) % D
    unsigned long long ans;
    mpz_t T;
    mpz_init_set_ui(T, X);
    mpz_mul_ui(T, T, Y);
    mpz_fdiv_r_ui(T, T, D);
    ans = mpz_get_ui(T);
    mpz_clear(T);
    return ans;
}

unsigned long long muldiv(unsigned long long X, unsigned long long Y, unsigned long long D)
{
    // return (X * Y) / D
    unsigned long long ans;
    mpz_t T;
    mpz_init_set_ui(T, X);
    mpz_mul_ui(T, T, Y);
    mpz_div_ui(T, T, D);
    ans = mpz_get_ui(T);
    mpz_clear(T);
    return ans;
}

bool is_2pow_1_mod_z(std::vector<unsigned> &P_vector, unsigned b, mpz_t &D)
{ 
    // return true if 2^P == 1 (mod D)
    bool ans = false;
    long long len_V = P_vector.size();
    mpz_t x;
    mpz_init_set_ui(x, 1);
    mpz_mul_2exp(x, x, P_vector[len_V - 1]);
    if (len_V > 1)
    {
        while (len_V > 1)
        {
            len_V--;
            for (unsigned j = 1; j < b; j++) 
            {
                mpz_mul(x, x, x);
                mpz_fdiv_r(x, x, D);
            }
            mpz_mul(x, x, x);
            mpz_mul_2exp(x, x, P_vector[len_V - 1]);
            mpz_fdiv_r(x, x, D);
        }
    }
    else
        mpz_fdiv_r(x, x, D);

    if (mpz_cmp_ui(x, 1) == 0)
        ans = true;
    mpz_clear(x);
        
    return ans;
}

int MpTF(unsigned long long P, int min_exp_D , int max_exp_D)
{ 
    // for P >= 127 if existing, it returns the factors D of (2^P - 1) with  2^min_exp_D < D < 2^max_exp_D 
    
    long long p_sieve_max = 524288; // maximum value small primes for sieve

    //construction of vector P in base 2^b
    const unsigned b = 6;
    std::vector<unsigned> P_vector;
    if (P >= (1ull << b))
    {
        unsigned long long P_temp = P;
        unsigned base_temp_m1 = (1 << b) - 1;
        while (P_temp > 0)
        {
            P_vector.push_back(P_temp & base_temp_m1);
            P_temp >>= b;
        }
    }
    else
        P_vector.push_back(P);
        
    min_exp_D = std::max(1, min_exp_D);
    long long PrimesBase[3] = {5 , 7 , 11};
    long long bW = 24;
    int n_PB = 1;
    if (n_PB_v > 1 && n_PB_v < 4)
        n_PB = n_PB_v;
    for(int j = 0; j < n_PB; j++)
        bW *= PrimesBase[j];
    unsigned long long p_max = (1ull << 63) / bW;
    if (P >= 127 && P < p_max && (min_exp_D < max_exp_D))
    {
        bW *= P;//modulus bW = 8 * 3 * p_1 * ...* p_(n_PB) * P  
        // vector for store useful residue classes       
        std::vector<long long> RW;
        long long p0_sieve = PrimesBase[n_PB - 1] + 2;        
        long long r_t1 , r_t7;
        int j1;
        r_t1 = 1;        
        r_t7 = 1 + 2 * P;
        if (P % 4 == 1)
            r_t7 += 4 * P;
        for (long long k = 1; k <= (long long)( bW / 8 / P); k++)
        {
            for (j1 = 0; j1 < n_PB; j1++)
                if (r_t1 % PrimesBase[j1] == 0 || r_t1 % 3 == 0)
                    break;
            if (j1 == n_PB)
                RW.push_back(r_t1);
            for (j1 = 0 ; j1 < n_PB ; j1++)
                if (r_t7 % PrimesBase[j1] == 0 || r_t7 % 3 == 0)
                    break;
            if (j1 == n_PB)
                RW.push_back(r_t7);
            r_t1 += 8 * P;
            r_t7 += 8 * P;
        }
        int  nR = RW.size(); //nB number of usefulr residue classes nR=phi(bW)
        int nB = nR / 8; //nB number of byte for nR residue classes
            
        unsigned long long  m;
        // set the maximum value small primes for sieve
        long long  dim_primes = std::min(bW , p_sieve_max);    
        // vector for find small primes for sieve
        std::vector<char> Primes_s(dim_primes, true); 
        // find small primes for sieve    
        for (long long p = 3; p * p < dim_primes ; p += 2)
            if (Primes_s[p])
                for (m = p * p; m < (unsigned long long)dim_primes; m += 2 * p)
                    Primes_s[m] = false;
        for (long long p = P; p < dim_primes ; p += P)
            Primes_s[p] = false;
        for(int j = 1; j < nR; j++)
            if (RW[j] < dim_primes)
                Primes_s[RW[j]] = false;        

        unsigned long long dim_seg = nB * k_segment_size;            
        // vector for find factor primes remainder RWi mod bW  p_k= RWi+bW*k  for  k>=0
        std::vector<uint8_t> Factor_k(dim_seg, 0xff); 

        int count_f = 0;
        long long r_t;
        unsigned long long num_seg = 1;
        mpz_t k_low, k_hi;
        mpz_t v_temp;
        mpz_init(v_temp);
        mpz_init(k_low);
        mpz_init(k_hi);
        mpz_setbit(k_low, min_exp_D);
        mpz_fdiv_q_ui(k_low, k_low, bW);
        mpz_setbit(k_hi, max_exp_D);
        mpz_fdiv_q_ui(k_hi, k_hi, bW);
        mpz_sub(v_temp, k_hi, k_low);
        if (mpz_cmp_ui(v_temp, k_segment_size) >= 0)
        {
            mpz_fdiv_q_ui(v_temp, v_temp, k_segment_size);
            num_seg += mpz_get_ui(v_temp);
        }
        if (mpz_cmp_ui(k_low, 0) == 0)
            Factor_k[0] &= del_bit[0];
        unsigned long long k_max = k_segment_size;
        for (unsigned long long i_seg = 0; i_seg < num_seg ; i_seg++)
        {
            mpz_add_ui(v_temp, k_low, k_segment_size);
            if (mpz_cmp(v_temp, k_hi) > 0)
            {
                mpz_sub(v_temp, k_hi, k_low);
                k_max = mpz_get_ui(v_temp) + 1;
            }
            for (long long p = p0_sieve; p < dim_primes; p += 2)
            {
                
                if (Primes_s[p])
                {
                    r_t1 = modinv(p, bW);
                    if (r_t1 < 0)
                        r_t1 += bW;
                       
                    unsigned long long pb = p * nB;
                    for(int jb = 0; jb < nB; jb++)
                    {
                        for(int j = 0; j < 8; j++)
                        {
                            // delete multiples of small primes in Factor_k with residue RW[j]
                            r_t = mulmod(r_t1, RW[j + 8ull * jb], bW);
                            m = muldiv(p, r_t, bW);
                            
                            if (mpz_cmp_ui(k_low, m) >= 0)
                            {
                                mpz_sub_ui(v_temp, k_low, m);
                                mpz_fdiv_r_ui(v_temp, v_temp, p);
                                m = mpz_get_ui(v_temp);
                                m = (p - m) % p;
                            }
                            else
                            {
                                m = m - mpz_get_ui(k_low);
                            }
                            if (m == (unsigned long long)p)
                                m = 0;
                            if (m < k_segment_size)
                            {
                                m *= nB;
                                m += jb;
                                for (; m < dim_seg; m += pb)
                                    Factor_k[m] &= del_bit[j];
                            }
                        }
                    }
                }
            }
            #pragma omp parallel for
            for (unsigned long long k = 0; k < k_max; k++)
            {
                mpz_t d_temp;
                mpz_init(d_temp);
                for(int jb = 0; jb < nB; jb++)
                {
                    uint8_t B_t = Factor_k[k * nB + jb];
                    for(int j = 7; j >= 0; j--)
                    {
                        if (B_t & (1 << j))
                        {
                            mpz_add_ui(d_temp, k_low, k);
                            mpz_mul_ui(d_temp, d_temp, bW);
                            mpz_add_ui(d_temp, d_temp, RW[7 - j + 8 * jb] );
                            if (is_2pow_1_mod_z(P_vector, b, d_temp))
                            {
                                if(count_f == 0)
                                {
                                    time_t time_c;
                                    char datetimestr[20];
                                    struct tm * datetime;
                                    time(&time_c);
                                    datetime = localtime(&time_c);
                                    strftime(datetimestr, 20, "%F %T", datetime);
                                    std::cout << "\n{\"timestamp\":\"" << datetimestr << "\", \"exponent\":" << P << ", \"worktype\":\"TF\", \"status\":\"F\", ";
                                    std::cout << "\"bitlo\":" << min_exp_D << ", \"bithi\":" << max_exp_D << ", \"rangecomplete\":true, \"factors\":[";
                                }
                                else
                                {
                                    std::cout << ",";
                                }                                    
                                std::cout << "\"" <<mpz_get_str(NULL, 10, d_temp) << "\"";
                                count_f++;
                            }
                        }
                    }
                }
                mpz_clear(d_temp);
            }
                
            std::fill(Factor_k.begin(), Factor_k.end(), 0xff);
            mpz_add_ui(k_low, k_low, k_segment_size);
        }
        //clear
        mpz_clear(k_low);
        mpz_clear(k_hi);
        mpz_clear(v_temp);
        return count_f; 
    }
    else
        std::cout << "\n MpTF(P,min_exp_D,max_exp_D) - enter P >= 127 and p < " << p_max <<" and max_exp_D > min_exp_D" << std::endl;
    return -1;
}

unsigned long long first_factor(unsigned long long n)
{
    //return 1 if n prime number else return first factor > 1
    unsigned long long q = (n % 2 == 0 && n > 2) ? 2 : ((n % 3 == 0 && n > 3) ? 3 : ((n % 5 == 0 && n > 5) ? 5 : ((n % 7 == 0 && n > 7) ? 7 : ((n > 1) ? 1 : 0))));
    if (n >= 121)
    {
        int word_exp = 3;
        int word_m1 = (1 << word_exp) - 1;
        const uint8_t del_bit[8] = {0x7f,0xbf,0xdf,0xef,0xf7,0xfb,0xfd,0xfe};
        unsigned long long sqrt_n = (unsigned long long) std::sqrt(n);
        unsigned long long modulus = 210;
        unsigned long long nR = 48;
        unsigned long long nB = nR >> word_exp;    
        unsigned long long R[nR] = {11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,121,127,131,137,139,143,149,151,157,163,167,169,173,179,181,187,191,193,197,199,209,211};
        unsigned long long Ri[nR] = {191,97,173,199,137,29,61,193,41,127,143,107,89,31,163,71,187,109,167,59,13,131,157,53,79,197,151,43,101,23,139,47,179,121,103,67,83,169,17,149,181,73,11,37,113,19,209,211};
        unsigned long long k_sqrt = (sqrt_n / modulus) + 1;
        unsigned long long k_sqrt_sqrt = (unsigned long long) std::sqrt(k_sqrt / modulus) + 1;
        unsigned long long vector_size = (k_sqrt + 1) * nB;
        std::vector<uint8_t> Primes(vector_size, 0xff);
        unsigned long long  k, i, ip, jp, mb, pb, rt;
        k = 0;
        while (q == 1 && k <= k_sqrt_sqrt)
        {
            for (jp = 0; jp < nR; jp++)
            {
                if(Primes[k * nB + (jp >> word_exp)] & (1 << (word_m1 - (jp & word_m1))))
                {
                    pb = modulus * k + R[jp];
                    if (n > pb && n % pb == 0)
                    {
                        q = pb;
                        break;
                    }
                    pb *= nB;
                    for (ip = 0ull; ip < nR; ip++)
                    {
                        rt = (R[ip] * Ri[jp]) % modulus;
                        mb = (ip >> word_exp) + nB * (rt * k + (R[jp] * rt) / modulus) + k * pb;
                        if (ip == nR - 1)
                            mb -= nB;
                        i = ip & word_m1;
                        for (; mb < vector_size; mb += pb)
                            Primes[mb] &= del_bit[i];
                    }
                }
            }
            k++;
        }
        while (q == 1 && k <= k_sqrt)
        {
            for (jp = 0; jp < nR; jp++)
            {
                if(Primes[k * nB + (jp >> word_exp)] & (1 << (word_m1 - (jp & word_m1))))
                {
                    pb = modulus * k + R[jp];
                    if (n > pb && n % pb == 0)
                    {
                        q = pb;
                        break;
                    }
                }
            }
            k++;
        }
    }
    return q;
}

int main(int argc, char *argw[])
{
    char username[] = "User140242"; //modify with your own data
    char pcname[] = "unknown"; //modify with your own data
    if (argc == 4 || argc == 5)
    {
        unsigned long long p_start = strtoll(argw[1], NULL, 10);
        unsigned long long p_stop = p_start;
        int bit_i;
        int bit_f;
        if(argc == 4)
        {
            bit_i = (int)strtol(argw[2], NULL, 10);
            bit_f = (int)strtol(argw[3], NULL, 10);
        }
        else
        {
            p_start += 1 - p_start % 2;
            p_stop = strtoll(argw[2], NULL, 10);
            bit_i = (int)strtol(argw[3], NULL, 10);
            bit_f = (int)strtol(argw[4], NULL, 10);
        }    
        bit_i = std::max(1, bit_i);
        bit_i = std::min(127, bit_i);
        bit_f = std::min(127, bit_f);
        time_t time_c;
        char datetimestr[20];
        struct tm * datetime;
        unsigned long long q, p;
        auto ti = std::chrono::system_clock::now();
        for (p = p_start; p <= p_stop; p += 2)
        {            
            int count_f = 0;    
            q = first_factor(p);
            if (q == 1)
            {
                count_f = MpTF(p , bit_i , bit_f);
                if (count_f == -1)
                {
                    std::cout << "\n error enter p >= 127 or the p value is too large" << std::endl;
                    break;
                }
                else
                {
                    if (count_f > 0)
                        std::cout << "], \"program\":{\"name\":\"MpTF\",\"version\":\"1.0.3\"}, \"user\":\"" << username << "\", \"computer\":\"" << pcname << "\"}";
                    else
                    {
                        time(&time_c);
                        datetime = localtime(&time_c);
                        strftime(datetimestr, 20, "%F %T", datetime);
                        std::cout << "\n{\"timestamp\":\"" << datetimestr << "\", \"exponent\":" << p << ", \"worktype\":\"TF\", \"status\":\"NF\", \"bitlo\":" << bit_i << ", \"bithi\":" << bit_f;
                        std::cout << ", \"rangecomplete\":true, \"program\":{\"name\":\"MpTF\",\"version\":\"1.0.3\"}, \"user\":\"" << username << "\", \"computer\":\"" << pcname << "\"}";
                    }
                    if (argc == 4)
                    {
                        std::cout << "\n \nexecuted trial factoring from 2^" << bit_i << " to 2^" << bit_f << std::endl;
                        if (count_f > 0)
                            std::cout << "found " << count_f << " factors" << std::endl;
                        else
                            std::cout << "\nNo factor found\n" << std::endl;
                    }
                }
            }
            else if (argc == 4)
                std::cout << p <<" is a composite number 2^" << q << "-1 is a factor of 2^" << p << "-1" << std::endl;
        }
        auto tf = std::chrono::system_clock::now();
        std::chrono::duration<double> delta_t = tf - ti;
        std::cout << "\n \n" << "t: " << delta_t.count() << " s" << std::endl;
    }
    else
    {
        std::cout << "call with:  " << argw[0] << " p bit_lo bit_hi \n(for trial factoring of 2^p-1 from 2^bit_lo to 2^bit_hi)" << std::endl;
        std::cout << "optional call with:  " << argw[0] << " p_start p_stop bit_lo bit_hi \n" << std::endl;
    }
    return 0;
}
