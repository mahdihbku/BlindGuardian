#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <openssl/obj_mac.h>
#include <openssl/ec.h>
#include <openssl/rand.h>
#include <openssl/bn.h>

BN_CTX *ctx;
EC_GROUP *curve;
const size_t priv_len = 32;
const size_t pub_len = 65;
const size_t POINT_UNCOMPRESSED_len = 65;	// z||x||y 		z is the octet 0x04
const size_t POINT_COMPRESSED_len = 33;		// z||x			z specifies which solution of the quadratic equation y is (0x02 or 0x03)
EC_POINT *h, *g;	//h: pub_key
BIGNUM *x, *q;		//x: priv_key
unsigned char* decryption_buf;

double print_time(struct timeval *start, struct timeval *end) {
	double usec;
	usec = (end->tv_sec*1000000 + end->tv_usec) - (start->tv_sec*1000000 + start->tv_usec);
	return usec/1000.0;
}

int prepare(char *pub_filename, char *priv_filename) {
	// seed PRNG
	if (RAND_load_file("/dev/urandom", 256) < 64) {
		printf("Can't seed PRNG!\n");
		abort();
	}
	ctx = BN_CTX_new();
	curve = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
	g = EC_POINT_new(curve);
	q = BN_new();
	g = EC_GROUP_get0_generator(curve);
	EC_GROUP_get_order(curve, q, ctx);
	x = BN_new();
	h = EC_POINT_new(curve);

	FILE *pub_file_p = fopen(pub_filename, "rb");
	unsigned char *pub_buf = malloc(sizeof(char)*POINT_UNCOMPRESSED_len);
	fread(pub_buf, POINT_UNCOMPRESSED_len, 1, pub_file_p);
	fclose(pub_file_p);
	FILE *priv_file_p = fopen(priv_filename, "rb");
	unsigned char *priv_buf = malloc(sizeof(char)*priv_len);
	fread(priv_buf, priv_len, 1, priv_file_p);
	fclose(priv_file_p);

	BN_bin2bn(priv_buf, priv_len, x);
	EC_POINT_oct2point(curve, h, pub_buf, POINT_UNCOMPRESSED_len, ctx);

	return 0;
}

int prepare_for_enc(char *pub_filename) {
	// seed PRNG
	if (RAND_load_file("/dev/urandom", 256) < 64) {
		printf("Can't seed PRNG!\n");
		abort();
	}
	ctx = BN_CTX_new();
	curve = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
	g = EC_POINT_new(curve);
	q = BN_new();
	g = EC_GROUP_get0_generator(curve);
	EC_GROUP_get_order(curve, q, ctx);
	h = EC_POINT_new(curve);

	// char *pub_filename = "ec_pub.txt";
	FILE *pub_file_p = fopen(pub_filename, "rb");
	unsigned char *pub_buf = malloc(sizeof(char)*pub_len);
	fread(pub_buf, pub_len, 1, pub_file_p);
	fclose(pub_file_p);

	EC_POINT_oct2point(curve, h, pub_buf, pub_len, ctx);

	return 0;
}

void encrypt_ec(char *cipherText, char *mess) {
	EC_POINT *c1, *c2;
	BIGNUM *m, *r;
	c2 = EC_POINT_new(curve);
	c1 = EC_POINT_new(curve);
	m = BN_new();
	r = BN_new();
	BN_dec2bn(&m, mess);
	BN_rand_range(r, q);
	EC_POINT_mul(curve, c1, NULL, g, r, ctx);
	BN_add(r, r, m);
	EC_POINT_mul(curve, c2, NULL, h, r, ctx);
    if (!EC_POINT_point2oct(curve, c1, POINT_CONVERSION_UNCOMPRESSED, cipherText, POINT_UNCOMPRESSED_len, ctx))
		printf("encrypt:error!\n");
	if (!EC_POINT_point2oct(curve, c2, POINT_CONVERSION_UNCOMPRESSED, cipherText+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx))
		printf("encrypt:error!\n");
}

void load_encryption_file() {
	FILE *dec_file_p = fopen("decrypt_file.bin", "rb");
	size_t file_size = (size_t)1073741824 * 17 * 2;
	decryption_buf = malloc(file_size * sizeof(unsigned char));
	fread(decryption_buf, file_size, 1, dec_file_p);
	fclose(dec_file_p);
}

int score_is_positive(char *ciphert) {
	EC_POINT *c1, *c2, *h1;
	BIGNUM *m;
	c1 = EC_POINT_new(curve);
	c2 = EC_POINT_new(curve);
	h1 = EC_POINT_new(curve);
	m = BN_new();
	EC_POINT_oct2point(curve, c1, ciphert, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c2, ciphert+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_mul(curve, c1, NULL, c1, x, ctx);
	EC_POINT_invert(curve, c1, ctx);
	EC_POINT_add(curve, c1, c1, c2, ctx);

	unsigned char* buf = malloc(POINT_COMPRESSED_len * sizeof(unsigned char));
	size_t c1_size = EC_POINT_point2oct(curve, c1, POINT_CONVERSION_COMPRESSED, buf, POINT_COMPRESSED_len, ctx);
	unsigned char x_buf[4];
	x_buf[0] = buf[1];	// these values gave the least collisions
	x_buf[1] = buf[3];
	x_buf[2] = buf[11];
	x_buf[3] = buf[14];
	size_t index = (unsigned long long)*((unsigned int*) x_buf);	// 4 bytes to be used as address in the hashing table
	index = index >> 2;	// index is 30 bits not 32
	index = index * 2 * 17;
	while (decryption_buf[index] != 0x0 && (decryption_buf[index] != buf[0] || memcmp(&decryption_buf[index+1], &buf[5], 16)))
		index+=17;
	if (decryption_buf[index] == 0x0)	// not found in table => negative
		return 0;
	else	// exists in table => positive
		return 1;
}

void decrypt_ec(char *message, char *ciphert) {
	EC_POINT *c1, *c2, *h1;
	BIGNUM *m;
	c1 = EC_POINT_new(curve);
	c2 = EC_POINT_new(curve);
	h1 = EC_POINT_new(curve);
	m = BN_new();
	EC_POINT_oct2point(curve, c1, ciphert, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c2, ciphert+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);

	EC_POINT_mul(curve, c1, NULL, c1, x, ctx);
	EC_POINT_invert(curve, c1, ctx);
	EC_POINT_add(curve, c1, c1, c2, ctx);
	EC_POINT_set_to_infinity(curve, h1);
	int i = 0;
	for (; i < 1000000; i++) {	// can decrypt only values that are < 1000000
		if (!EC_POINT_cmp(curve, h1, c1, ctx))
			break;
		EC_POINT_add(curve, h1, h1, h, ctx);
	}
	snprintf(message, 30, "%d", i);
}

void generate_decrypt_file() {
	unsigned char* buf = malloc(POINT_COMPRESSED_len * sizeof(unsigned char));
	unsigned char x_buf[4];
	size_t index, h1_size, collisions = 0, collision_moves, max_collision_moves=0, file_size = (size_t)1073741824 * 17 * 2;	//hash[0] in buf[0] for the sign, and hash[5..20] in buf[1..16] for the hash
	EC_POINT *h1 = EC_POINT_new(curve);
	EC_POINT_set_to_infinity(curve, h1);
	EC_POINT_add(curve, h1, h1, h, ctx);

	decryption_buf = (unsigned char*) calloc (file_size, sizeof(unsigned char));
	// Each ct is of size 32 bytes. 4 bytes will be used as @ => and 16+1 bytes as data. Bucket size = 2
	if (decryption_buf == NULL) {
		printf("generate_decrypt_file: ERROR: malloc failed\n");
		exit(0);
	}
	for (unsigned int i=0; i<1073741824; i++) {
		h1_size = EC_POINT_point2oct(curve, h1, POINT_CONVERSION_COMPRESSED, buf, POINT_COMPRESSED_len, ctx);
		x_buf[0] = buf[1];	// these values gave the least collisions
		x_buf[1] = buf[3];
		x_buf[2] = buf[11];
		x_buf[3] = buf[14];
		index = (unsigned long long)*((unsigned int*) x_buf);
		index = index >> 2;	// index is 30 bits not 32
		index = index * 2 * 17;	// 2 entries per bucket
		collision_moves = 0;
		while (decryption_buf[index] != 0x0) {	// not empty entries, if not empty it must be 0x2 or 0x3
			index+=17;
			collisions++;
			collision_moves++;
		}
		if (max_collision_moves < collision_moves) {
			max_collision_moves = collision_moves;
			printf("generate_decrypt_file: at index=%lu, collisions=%lu, max_collision_moves=%lu, i=%u/%u, table occupation=%lu%%\n", index, collisions, collision_moves, i/1024, 1073741824/1024, (size_t)i*100/1073741824);
		}
		decryption_buf[index] = buf[0];
		memcpy(&decryption_buf[index+1], &buf[5], 16);
		EC_POINT_add(curve, h1, h1, h, ctx);
	}
	printf("generate_decrypt_file: writing to file...\n");
	FILE *pfile;	// writing decryption_buf to file
	pfile = fopen("decrypt_file.bin", "wb");
	fwrite(decryption_buf, sizeof(unsigned char), file_size, pfile);
	fclose(pfile);
	printf("generate_decrypt_file: file generated successfully!\n");
}

void add2(char *result, char *ct1, char *ct2) {
	EC_POINT *c1, *c2, *c3, *c4;
	c1 = EC_POINT_new(curve);
	c2 = EC_POINT_new(curve);
	c3 = EC_POINT_new(curve);
	c4 = EC_POINT_new(curve);
	EC_POINT_oct2point(curve, c1, ct1, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c2, ct1+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c3, ct2, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c4, ct2+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);

	EC_POINT_add(curve, c1, c1, c3, ctx);
	EC_POINT_add(curve, c2, c2, c4, ctx);

    if (!EC_POINT_point2oct(curve, c1, POINT_CONVERSION_UNCOMPRESSED, result, POINT_UNCOMPRESSED_len, ctx))
		printf("add2:error!\n");
	if (!EC_POINT_point2oct(curve, c2, POINT_CONVERSION_UNCOMPRESSED, result+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx))
		printf("add2:error!\n");
}

void add3(char *result, char *ct1, char *ct2, char *ct3) {
	EC_POINT *c11, *c12, *c21, *c22, *c31, *c32;
	c11 = EC_POINT_new(curve);
	c12 = EC_POINT_new(curve);
	c21 = EC_POINT_new(curve);
	c22 = EC_POINT_new(curve);
	c31 = EC_POINT_new(curve);
	c32 = EC_POINT_new(curve);
	EC_POINT_oct2point(curve, c11, ct1, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c12, ct1+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c21, ct2, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c22, ct2+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c31, ct3, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c32, ct3+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);

	EC_POINT_add(curve, c11, c11, c21, ctx);
	EC_POINT_add(curve, c11, c11, c31, ctx);
	EC_POINT_add(curve, c12, c12, c22, ctx);
	EC_POINT_add(curve, c12, c12, c32, ctx);

	if (!EC_POINT_point2oct(curve, c11, POINT_CONVERSION_UNCOMPRESSED, result, POINT_UNCOMPRESSED_len, ctx))
		printf("add3:error!\n");
	if (!EC_POINT_point2oct(curve, c12, POINT_CONVERSION_UNCOMPRESSED, result+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx))
		printf("add3:error!\n");
}

void add4(char *result, char *ct1, char *ct2, char *ct3, char *ct4) {
	EC_POINT *c11, *c12, *c21, *c22, *c31, *c32, *c41, *c42;
	c11 = EC_POINT_new(curve);
	c12 = EC_POINT_new(curve);
	c21 = EC_POINT_new(curve);
	c22 = EC_POINT_new(curve);
	c31 = EC_POINT_new(curve);
	c32 = EC_POINT_new(curve);
	c41 = EC_POINT_new(curve);
	c42 = EC_POINT_new(curve);
	EC_POINT_oct2point(curve, c11, ct1, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c12, ct1+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c21, ct2, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c22, ct2+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c31, ct3, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c32, ct3+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c41, ct4, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c42, ct4+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);

	EC_POINT_add(curve, c11, c11, c21, ctx);
	EC_POINT_add(curve, c11, c11, c31, ctx);
	EC_POINT_add(curve, c11, c11, c41, ctx);
	EC_POINT_add(curve, c12, c12, c22, ctx);
	EC_POINT_add(curve, c12, c12, c32, ctx);
	EC_POINT_add(curve, c12, c12, c42, ctx);

	if (!EC_POINT_point2oct(curve, c11, POINT_CONVERSION_UNCOMPRESSED, result, POINT_UNCOMPRESSED_len, ctx))
		printf("add4:error!\n");
	if (!EC_POINT_point2oct(curve, c12, POINT_CONVERSION_UNCOMPRESSED, result+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx))
		printf("add4:error!\n");
}

void mult(char *result, char *scalar, char *ct1) {
	EC_POINT *c1, *c2;
	BIGNUM *m;
	c1 = EC_POINT_new(curve);
	c2 = EC_POINT_new(curve);
	m = BN_new();
	EC_POINT_oct2point(curve, c1, ct1, POINT_UNCOMPRESSED_len, ctx);
	EC_POINT_oct2point(curve, c2, ct1+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx);
	BN_dec2bn(&m, scalar);

	EC_POINT_mul(curve, c1, NULL, c1, m, ctx);
	EC_POINT_mul(curve, c2, NULL, c2, m, ctx);

	if (!EC_POINT_point2oct(curve, c1, POINT_CONVERSION_UNCOMPRESSED, result, POINT_UNCOMPRESSED_len, ctx))
		printf("add2:error!\n");
	if (!EC_POINT_point2oct(curve, c2, POINT_CONVERSION_UNCOMPRESSED, result+POINT_UNCOMPRESSED_len, POINT_UNCOMPRESSED_len, ctx))
		printf("add2:error!\n");
}

void generate_keys(char *pub_filename, char *priv_filename) {
	BN_CTX *ctx;
	EC_GROUP *curve;
	BIGNUM *x;
	EC_POINT *g, *h;
	EC_KEY *key;
	if (RAND_load_file("/dev/urandom", 256) < 64) {
		printf("Can't seed PRNG!\n");
		abort();
	}
	ctx = BN_CTX_new();
	x = BN_new();
	// create a standard 256-bit prime curve
	curve = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
	g = EC_POINT_new(curve);
	h = EC_POINT_new(curve);
	g = EC_GROUP_get0_generator(curve);
	key = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
	EC_KEY_generate_key(key);
	x = EC_KEY_get0_private_key(key);
	h = EC_KEY_get0_public_key(key);
	unsigned char *pub_buf = malloc(sizeof(char)*pub_len);
    if (EC_POINT_point2oct(curve, h, POINT_CONVERSION_UNCOMPRESSED, pub_buf, POINT_UNCOMPRESSED_len, ctx) != POINT_UNCOMPRESSED_len) {
		printf("error!");
		exit(0);
	}
	size_t priv_len = BN_num_bytes(x);
	unsigned char *priv_buf = malloc(sizeof(char)*priv_len);
	BN_bn2binpad(x, priv_buf, priv_len);
	FILE *pub_file_p = fopen(pub_filename, "wb");
	fwrite(pub_buf, pub_len, 1, pub_file_p);
	fclose(pub_file_p);
	FILE *priv_file_p = fopen(priv_filename, "wb");
	fwrite(priv_buf, priv_len, 1, priv_file_p);
	fclose(priv_file_p);
}

void test() {
	prepare("ec_pub.txt", "ec_priv.txt");
	// generate_decrypt_file();
	load_encryption_file();
	char* ct1 = (char*) malloc (130 * sizeof(char));
	encrypt_ec(ct1, "1234");
	char pt1[20];
	decrypt_ec(pt1, ct1);
	printf("pt1=%s\n", pt1);
	printf("score_is_positive(ct1)=%d\n", score_is_positive(ct1));

	char* ct2 = (char*) malloc (130 * sizeof(char));
	encrypt_ec(ct2, "-1234");
	char pt2[20];
	decrypt_ec(pt2, ct2);
	printf("pt2=%s\n", pt2);
	printf("score_is_positive(ct2)=%d\n", score_is_positive(ct2));


	encrypt_ec(ct1, "1284734");
	decrypt_ec(pt1, ct1);
	printf("pt1=%s\n", pt1);
	printf("score_is_positive(ct1)=%d\n", score_is_positive(ct1));


	encrypt_ec(ct2, "-128734");
	decrypt_ec(pt2, ct2);
	printf("pt2=%s\n", pt2);
	printf("score_is_positive(ct2)=%d\n", score_is_positive(ct2));
}

int main() {
	
}