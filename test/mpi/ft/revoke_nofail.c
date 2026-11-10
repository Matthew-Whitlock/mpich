/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int sbuf[2], rbuf[2];
MPI_Message msg;
MPI_Comm comm_output;
MPI_Group comm_group_tmp = MPI_GROUP_NULL;
typedef void (*revoke_test_fn)(int rank, MPI_Comm c);
typedef int (*test_fn)(MPI_Comm c);
typedef int (*test_ifn)(MPI_Comm c, MPI_Request * r);

static int comm_rank(MPI_Comm c) { int r; MPI_Comm_rank(c, &r); return r; }
static int comm_size(MPI_Comm c) { int s; MPI_Comm_size(c, &s); return s; }
static MPI_Group comm_group(MPI_Comm c) {
    if (comm_group_tmp != MPI_GROUP_NULL) MPI_Group_free(&comm_group_tmp);
    MPI_Comm_group(c, &comm_group_tmp);
    return comm_group_tmp;
}

static void verify_return_val(int rank, const char *test_type, const char *fname, int rc, int rc_expected)
{
    int errclass, resultlen;
    MPI_Error_class(rc, &errclass);
    if (errclass == rc_expected) return;
    char error[MPI_MAX_ERROR_STRING] = "", expected[MPI_MAX_ERROR_STRING] = "";
    MPI_Error_string(rc, error, &resultlen);
    MPI_Error_string(rc_expected, expected, &resultlen);
    
    fprintf(stderr, "[%d] %s: %s should have returned %d %s, but returned %d %s (%d)\n",
            rank, test_type, fname, rc_expected, expected, errclass, error, rc);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
static void verify_success(int rank, const char* test_type, const char* fname, int rc)
{
    verify_return_val(rank, test_type, fname, rc, MPI_SUCCESS);
}
static void verify_revoked(int rank, const char *test_type, const char *fname, int rc)
{
    verify_return_val(rank, test_type, fname, rc, MPIX_ERR_REVOKED);
}

static void test_fn_after_revoked(int rank, const char* fname, MPI_Comm c, test_fn fn)
{
    const char* test_type = "test_fn_after_revoked";
    if (rank == 1) MPIX_Comm_revoke(c);
    verify_revoked(rank, test_type, fname, fn(c));
}
static void test_fn_unmatched(int rank, const char* fname, MPI_Comm c, test_fn fn)
{
    const char* test_type = "test_fn_unmatched";
    if (rank == 1) MPIX_Comm_revoke(c);
    else verify_revoked(rank, test_type, fname, fn(c));
}
static void test_ifn_after_revoked(int rank, const char *fname, MPI_Comm c, test_ifn fn)
{
    const char* test_type = "test_ifn_after_revoked";
    if (rank == 1) MPIX_Comm_revoke(c);

    MPI_Request r = MPI_REQUEST_NULL;
    int rc = fn(c, &r);
    if (rc == MPI_SUCCESS) rc = MPI_Wait(&r, MPI_STATUS_IGNORE);

    verify_revoked(rank, test_type, fname, rc);
}
static void test_ifn_matched_after_revoked(int rank, const char *fname, MPI_Comm c, test_ifn fn)
{
    const char* test_type = "test_ifn_matched_after_revoked";
    MPI_Request r = MPI_REQUEST_NULL;
    if (rank != 1) verify_success(rank, test_type, fname, fn(c, &r));
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) MPIX_Comm_revoke(c);

    if (rank == 1) {
        verify_revoked(rank, test_type, fname, fn(c, &r));
    } else if (rank == 0) {
        verify_revoked(rank, test_type, fname, MPI_Wait(&r, MPI_STATUS_IGNORE));
    }
}
static void test_ifn_unmatched(int rank, const char *fname, MPI_Comm c, test_ifn fn)
{
    const char* test_type = "test_ifn_unmatched";
    MPI_Request r = MPI_REQUEST_NULL;
    if (rank != 1) verify_success(rank, test_type, fname, fn(c, &r));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) MPIX_Comm_revoke(c);

    if (rank != 1) {
        verify_revoked(rank, test_type, fname, MPI_Wait(&r, MPI_STATUS_IGNORE));
    }
}



#define BASIC_TEST_FNS(name)                                              \
    static void test_##name##_after_revoke(int rank, MPI_Comm c)          \
    {                                                                     \
        test_fn_after_revoked(rank, "MPI_" #name, c, name);               \
    }                                                                     \
    static void test_##name##_unmatched(int rank, MPI_Comm c)             \
    {                                                                     \
        test_fn_unmatched(rank, "MPI_" #name, c, name);                   \
    }
#define NB_TEST_FNS(name)                                                 \
    static void test_i##name##_after_revoke(int rank, MPI_Comm c)         \
    {                                                                     \
        test_ifn_after_revoked(rank, "MPI_I" #name, c, i##name);          \
    }                                                                     \
    static void test_i##name##_matched_after_revoke(int rank, MPI_Comm c) \
    {                                                                     \
        test_ifn_matched_after_revoked(rank, "MPI_I" #name, c, i##name);  \
    }                                                                     \
    static void test_i##name##_unmatched(int rank, MPI_Comm c)            \
    {                                                                     \
        test_ifn_unmatched(rank, "MPI_I" #name, c, i##name);              \
    }
#define FULL_TEST_FNS(name)                                               \
    BASIC_TEST_FNS(name)                                                  \
    NB_TEST_FNS(name)

#define BASIC_TEST_FN_NAMES(name)        \
    test_##name##_after_revoke,          \
    test_##name##_unmatched
#define NB_TEST_FN_NAMES(name)           \
    test_i##name##_after_revoke,         \
    test_i##name##_matched_after_revoke, \
    test_i##name##_unmatched
#define FULL_TEST_FN_NAMES(name)         \
    BASIC_TEST_FN_NAMES(name),           \
    NB_TEST_FN_NAMES(name)

static int barrier(MPI_Comm c){ return MPI_Barrier(c); }
static int ibarrier(MPI_Comm c, MPI_Request* r){ return MPI_Ibarrier(c, r); }
FULL_TEST_FNS(barrier)
static int bcast(MPI_Comm c){ return MPI_Bcast(sbuf, 1, MPI_INT, 1, c); }
static int ibcast(MPI_Comm c, MPI_Request * r){ return MPI_Ibcast(sbuf, 1, MPI_INT, 1, c, r); }
FULL_TEST_FNS(bcast)
static int scatter(MPI_Comm c){ return MPI_Scatter(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, 1, c); }
static int iscatter(MPI_Comm c, MPI_Request * r){ return MPI_Iscatter(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, 1, c, r); }
FULL_TEST_FNS(scatter)
static int reduce(MPI_Comm c){ return MPI_Reduce(sbuf, rbuf, 1, MPI_INT, MPI_SUM, 0, c); }
static int ireduce(MPI_Comm c, MPI_Request * r){ return MPI_Ireduce(sbuf, rbuf, 1, MPI_INT, MPI_SUM, 0, c, r); }
FULL_TEST_FNS(reduce)
static int gather(MPI_Comm c){ return MPI_Gather(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, 0, c); }
static int igather(MPI_Comm c, MPI_Request * r){ return MPI_Igather(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, 0, c, r); }
FULL_TEST_FNS(gather)
static int alltoall(MPI_Comm c){ return MPI_Alltoall(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, c); }
static int ialltoall(MPI_Comm c, MPI_Request * r){ return MPI_Ialltoall(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, c, r); }
FULL_TEST_FNS(alltoall)
static int allgather(MPI_Comm c){ return MPI_Allgather(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, c); }
static int iallgather(MPI_Comm c, MPI_Request * r){ return MPI_Iallgather(sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, c, r); }
FULL_TEST_FNS(allgather)
static int allreduce(MPI_Comm c){ return MPI_Allreduce(sbuf, rbuf, 1, MPI_INT, MPI_SUM, c); }
static int iallreduce(MPI_Comm c, MPI_Request * r){ return MPI_Iallreduce(sbuf, rbuf, 1, MPI_INT, MPI_SUM, c, r); }
FULL_TEST_FNS(allreduce)

static int recv(MPI_Comm c) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++){
            if (i == 1) continue;
            int op_err = MPI_Send(sbuf, 1, MPI_INT, i, 100, c);
            if (op_err && ! ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Recv(rbuf, 1, MPI_INT, 1, 100, c, MPI_STATUS_IGNORE);
    }
}
static int irecv(MPI_Comm c, MPI_Request* r) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++){
            if (i == 1) continue;
            // In the test-passing cases, it's fine to not fill the request on rank 1,
            // because this operation should always fail and the request should be meaningless.
            int op_err = MPI_Send(sbuf, 1, MPI_INT, i, 101, c);
            if (op_err && !ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Irecv(rbuf, 1, MPI_INT, 1, 101, c, r);
    }
}
FULL_TEST_FNS(recv)

static int recv_any_source(MPI_Comm c) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++) {
            if (i == 1) continue;
            int op_err = MPI_Send(sbuf, 1, MPI_INT, i, 104, c);
            if (op_err && !ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Recv(rbuf, 1, MPI_INT, MPI_ANY_SOURCE, 104, c, MPI_STATUS_IGNORE);
    }
}
static int irecv_any_source(MPI_Comm c, MPI_Request * r) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++) {
            if (i == 1) continue;
            /* In the test-passing cases, it's fine to not fill the request on rank 1,
             * because this operation should always fail and the request should be meaningless. */
            int op_err = MPI_Send(sbuf, 1, MPI_INT, i, 105, c);
            if (op_err && !ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Irecv(rbuf, 1, MPI_INT, MPI_ANY_SOURCE, 105, c, r);
    }
}
FULL_TEST_FNS(recv_any_source)

static int sendrecv(MPI_Comm c) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++){
            if (i == 1) continue;
            int op_err = MPI_Sendrecv(sbuf, 1, MPI_INT, i, 102,
                                      rbuf, 1, MPI_INT, i, 102, c, MPI_STATUS_IGNORE);
            if (op_err && ! ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Sendrecv(sbuf, 1, MPI_INT, 1, 102,
                            rbuf, 1, MPI_INT, 1, 102, c, MPI_STATUS_IGNORE);
    }
}
static int isendrecv(MPI_Comm c, MPI_Request* r) {
    if (comm_rank(c) == 1) {
        int ret_err = MPI_SUCCESS;
        for (int i = 0; i < comm_size(c); i++){
            if (i == 1) continue;
            // In the test-passing cases, it's fine to reuse the same request on rank 1,
            // because this operation should always fail and the request should be meaningless.
            int op_err = MPI_Isendrecv(sbuf, 1, MPI_INT, i, 103,
                                       rbuf, 1, MPI_INT, i, 103, c, r);
            if (op_err && ! ret_err) ret_err = op_err;
        }
        return ret_err;
    } else {
        return MPI_Isendrecv(sbuf, 1, MPI_INT, 1, 103,
                             rbuf, 1, MPI_INT, 1, 103, c, r);
    }
}
FULL_TEST_FNS(sendrecv)

static void test_probe_unmatched(int rank, MPI_Comm c)
{
    if (rank == 1) MPIX_Comm_revoke(c);
    int rc = MPI_Probe(1, 200, c, MPI_STATUS_IGNORE);
    verify_revoked(rank, "probe_unmatched", "MPI_Probe", rc);
}
static void test_iprobe_unmatched(int rank, MPI_Comm c)
{
    if (rank == 1) MPIX_Comm_revoke(c);
    int rc = MPI_SUCCESS, flag = false;
    while (rc == MPI_SUCCESS && !flag)
        rc = MPI_Iprobe(1, 201, c, &flag, MPI_STATUS_IGNORE);
    verify_revoked(rank, "iprobe_unmatched", "MPI_Iprobe", rc);
}
static void test_mprobe_unmatched(int rank, MPI_Comm c)
{
    if (rank == 1) MPIX_Comm_revoke(c);
    int rc = MPI_Mprobe(1, 202, c, &msg, MPI_STATUS_IGNORE);
    verify_revoked(rank, "mprobe_unmatched", "MPI_Mprobe", rc);
}
static void test_improbe_unmatched(int rank, MPI_Comm c)
{
    if (rank == 1) MPIX_Comm_revoke(c);
    int rc = MPI_SUCCESS, flag = false;
    while (rc == MPI_SUCCESS && !flag)
        rc = MPI_Improbe(1, 203, c, &flag, &msg, MPI_STATUS_IGNORE);
    verify_revoked(rank, "improbe_unmatched", "MPI_Improbe", rc);
}

static int comm_dup(MPI_Comm c){ return MPI_Comm_dup(c, &comm_output); }
static int icomm_dup(MPI_Comm c, MPI_Request *r){ return MPI_Comm_idup(c, &comm_output, r); }
FULL_TEST_FNS(comm_dup)

static int comm_create(MPI_Comm c){ return MPI_Comm_create(c, comm_group(c), &comm_output); }
BASIC_TEST_FNS(comm_create)

static int comm_create_group(MPI_Comm c){ return MPI_Comm_create_group(c, comm_group(c), 0, &comm_output); }
BASIC_TEST_FNS(comm_create_group)

static int comm_split(MPI_Comm c) { return MPI_Comm_split(c, 0, comm_rank(c), &comm_output); }
BASIC_TEST_FNS(comm_split)

int main(int argc, char **argv)
{
    static const revoke_test_fn tests[] = {
        FULL_TEST_FN_NAMES(barrier),
        FULL_TEST_FN_NAMES(bcast),
        FULL_TEST_FN_NAMES(scatter),
        FULL_TEST_FN_NAMES(reduce),
        FULL_TEST_FN_NAMES(gather),
        FULL_TEST_FN_NAMES(alltoall),
        FULL_TEST_FN_NAMES(allgather),
        FULL_TEST_FN_NAMES(allreduce),
        FULL_TEST_FN_NAMES(recv),
        FULL_TEST_FN_NAMES(recv_any_source),
        FULL_TEST_FN_NAMES(sendrecv),
        FULL_TEST_FN_NAMES(comm_dup),
        BASIC_TEST_FN_NAMES(comm_create),
        BASIC_TEST_FN_NAMES(comm_create_group),
        BASIC_TEST_FN_NAMES(comm_split),
        test_probe_unmatched, test_iprobe_unmatched,
        test_mprobe_unmatched, test_improbe_unmatched,
    };

    MPI_Init(&argc, &argv);
    int size = comm_size(MPI_COMM_WORLD);
    if (size < 2) {
        fprintf(stderr, "Must run with at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

        MPI_Barrier(MPI_COMM_WORLD);
        tests[i](comm_rank(comm), comm);

        MPI_Comm_free(&comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (comm_rank(MPI_COMM_WORLD) == 0) {
        fprintf(stdout, " No errors\n");
    }

    MPI_Finalize();

    return 0;
}
