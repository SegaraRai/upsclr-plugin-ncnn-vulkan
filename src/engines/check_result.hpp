#ifndef CHECK_RESULT_HPP
#define CHECK_RESULT_HPP

#define CHECK_NCNN_RESULT_AND_LOG(logger_error, expr)                      \
    if (const int code = expr; code != 0) {                                \
        logger_error->error("[{}] {} returned {}", __func__, #expr, code); \
        return code;                                                       \
    }

#endif
