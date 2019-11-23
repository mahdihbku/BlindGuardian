%module ec_elgamal
%include "cdata.i"
%{
/* Put header files here or function declarations like below */
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <openssl/obj_mac.h>
#include <openssl/ec.h>
#include <openssl/rand.h>
#include <openssl/bn.h>

extern BN_CTX *ctx;
extern EC_GROUP *curve;
extern EC_POINT *h, *g;		//h: pub_key
extern BIGNUM *x, *q;		//x: priv_key

extern int prepare(char *pub, char *priv);
extern int prepare_for_enc(char *pub);
extern double print_time(struct timeval *start, struct timeval *end);
extern void generate_decrypt_file();
extern void load_encryption_file();
extern void generate_keys(char *pub_filename, char *priv_filename);
extern int score_is_positive(char *ciphert);
extern void decrypt_ec(char *message, char *ciphert);
extern void encrypt_ec(char *cipherText, char *mess);
extern void decrypt_ec_time_only(char *message, char *ciphert);
extern void add2(char *result, char *ct1, char *ct2);
extern void add3(char *result, char *ct1, char *ct2, char *ct3);
extern void add4(char *result, char *ct1, char *ct2, char *ct3, char *ct4);
extern void mult(char *result, char *scalar, char *ct1);
%}

%include "typemaps.i"
%include "cstring.i"

extern int prepare(char *pub, char *priv);
extern int prepare_for_enc(char *pub);
extern double print_time(struct timeval *start, struct timeval *end);
extern void generate_decrypt_file();
extern void load_encryption_file();
extern int score_is_positive(char *ciphert);
extern void generate_keys(char *pub_filename, char *priv_filename);

%cstring_bounded_output(char *message, 30);
extern void decrypt_ec(char *message, char *ciphert);

%cstring_chunk_output(char *cipherText, 130);
extern void encrypt_ec(char *cipherText, char *mess);

%cstring_chunk_output(char *result, 130);
extern void add2(char *result, char *ct1, char *ct2);

%cstring_chunk_output(char *result, 130);
extern void add3(char *result, char *ct1, char *ct2, char *ct3);

%cstring_chunk_output(char *result, 130);
extern void add4(char *result, char *ct1, char *ct2, char *ct3, char *ct4);

%cstring_chunk_output(char *result, 130);
extern void mult(char *result, char *scalar, char *ct1);