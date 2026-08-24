// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "AICMetadata.h"
#include "metadataFlatbufferWriter.h"

#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Device-config constants (mirrors software-kit runtime/lib/AICDefsInternal.h
// and lib/program/Program.cpp).
// ---------------------------------------------------------------------------
namespace {
constexpr int MAX_NUM_CORES = 16;
constexpr int MAX_NUM_THREADS = 6;
constexpr uint32_t CACHE_LINE_SIZE = 128;
constexpr uint32_t NUM_UDMA_CACHELINES_PER_THREAD = 2;

constexpr uint32_t VTCM_MAX_SIZE = 8 * 1024 * 1024;  // 8MB VTCM
constexpr uint32_t L2TCM_MAX_SIZE = 1 * 1024 * 1024; // 1MB L2TCM
constexpr uint32_t COMPUTE_MAX_SEMAPHORES = 2;
constexpr uint32_t DB_SIZE = 4; // bytes per doorbell

// Matches struct __attribute__((packed, aligned(16))) aic::DMADescriptor:
// four uint32_t words = 16 bytes.
constexpr uint32_t DMA_DESCRIPTOR_SIZE = 16;

static uint64_t alignTo(uint64_t x, uint64_t m) {
  return ((x + (m - 1)) & ~(m - 1)); // Only works if alignment is 2^n
}

// Serialized program-description structs (mirrors runtime/lib BufferDesc.h and
// SerializedProgramDesc.h; kept byte-identical for constants.bin).
enum usageType_t : uint32_t {
  USAGE_INPUT = 0,
  USAGE_OUTPUT = 1,
  USAGE_INTERNAL = 2,
};

enum memLoc_t : uint32_t { MEM_L2TCM = 0, MEM_VTCM = 1, MEM_DDR = 2 };

struct BufferDesc_t {
  memLoc_t location;
  uint32_t offset;
  uint32_t size;
  uint16_t waitDBNum;
  uint16_t ioDBNum;
  uint32_t waitDBVal;
  uint32_t ioDBVal;
  uint16_t ioMCID;
  uint16_t ioDBMCID;
  uint16_t buffMCID;
  uint16_t nspMask;
  usageType_t usage;
  uint32_t allowPartial;
};
static_assert(sizeof(BufferDesc_t) == 40, "BufferDesc_t must be 40 bytes");

constexpr uint16_t SERIALIZED_PROGRAMDESC_VERSION = 1;
struct SerializedProgramDesc_t {
  uint16_t serialVersion;
  uint16_t exitDB;
  uint32_t size;
  uint32_t numThreads;
  uint16_t numBuffs;
  uint16_t numInputBuffs;
  uint16_t numOutputBuffs;
  uint16_t numInternalBuffs;
  uint16_t inputSem;
  uint16_t outputSem;
  uint16_t hasInputsMask;
  uint16_t hasOutputsMask;
  uint32_t buffersOffset;
  uint32_t udmaDescBuffNum;
  uint32_t udmaDummyStartDescOffset;
};

// Memory destination for an I/O buffer.
enum destination { DEST_DDR = 0, DEST_L2TCM = 1, DEST_VTCM = 2 };

// ---------------------------------------------------------------------------
// md_struct.bin layout (mirrors metadata_reader.py / the driver's
// qaic::metadata::CompactMetadata). The driver reads an 88-byte header
// followed by dma_req_count fixed-size DMA request records.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct MdDoorbell {
  uint32_t data;
  uint32_t mc_id;
  uint64_t offset;
  uint32_t size;
  uint32_t _pad;
};
static_assert(sizeof(MdDoorbell) == 24, "MdDoorbell must be 24 bytes");

struct MdSemaphore {
  uint32_t in_sync_fence;
  uint32_t out_sync_fence;
  uint32_t pre_or_post;
  uint32_t sem_num;
  uint32_t sem_op;
  uint32_t sem_value;
};
static_assert(sizeof(MdSemaphore) == 24, "MdSemaphore must be 24 bytes");

struct MdDmaRequest {
  uint32_t dev_addr_space;
  uint32_t num;
  uint64_t dev_addr_offset;
  uint64_t host_addr_offset;
  uint64_t size;
  uint32_t in_out;
  uint32_t mc_id;
  uint32_t port_id;
  uint32_t semaphore_count;
  MdDoorbell doorbell;
  MdSemaphore semaphore_ops[4];
};
static_assert(sizeof(MdDmaRequest) == 168, "MdDmaRequest must be 168 bytes");

struct MdHeader {
  char network_name[64];
  uint32_t nsp_count;
  uint32_t semaphores_count;
  uint32_t l2tcm_size;
  uint32_t vtcm_size;
  uint32_t dma_req_count;
  uint8_t static_constants_exist;
  uint8_t _pad[3];
};
static_assert(sizeof(MdHeader) == 88, "MdHeader must be 88 bytes");
#pragma pack(pop)

// ---------------------------------------------------------------------------
// Minimal JSON reader for jithwdev.json. The config schema is small and flat
// (object with scalar fields plus "inputs"/"outputs" arrays of flat objects);
// this parser covers that subset. jithwdev.json is subject to change, so the
// parser reads named fields rather than assuming a fixed layout.
// ---------------------------------------------------------------------------
class Json {
public:
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  double num = 0;
  bool boolean = false;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool has(const std::string &k) const {
    return type == Type::Object && obj.count(k);
  }
  const Json &at(const std::string &k) const {
    static const Json nullJson;
    auto it = obj.find(k);
    return it == obj.end() ? nullJson : it->second;
  }
  int64_t asInt(int64_t dflt = 0) const {
    if (type == Type::Number)
      return (int64_t)num;
    if (type == Type::String) {
      // Allow numeric strings.
      try {
        return (int64_t)std::stoll(str);
      } catch (...) {
        return dflt;
      }
    }
    if (type == Type::Bool)
      return boolean ? 1 : 0;
    return dflt;
  }
  bool asBool(bool dflt = false) const {
    if (type == Type::Bool)
      return boolean;
    if (type == Type::Number)
      return num != 0;
    if (type == Type::String)
      return str == "true" || str == "1";
    return dflt;
  }
  std::string asStr(const std::string &dflt = "") const {
    return type == Type::String ? str : dflt;
  }
};

class JsonParser {
public:
  explicit JsonParser(const std::string &text) : s(text), i(0) {}
  Json parse() {
    skipWs();
    Json v = parseValue();
    return v;
  }

private:
  const std::string &s;
  size_t i;

  [[noreturn]] void fail(const std::string &msg) {
    fprintf(stderr, "mdgen: JSON parse error at offset %zu: %s\n", i,
            msg.c_str());
    exit(1);
  }
  void skipWs() {
    while (i < s.size()) {
      char c = s[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        i++;
      else
        break;
    }
  }
  Json parseValue() {
    skipWs();
    if (i >= s.size())
      fail("unexpected end of input");
    char c = s[i];
    if (c == '{')
      return parseObject();
    if (c == '[')
      return parseArray();
    if (c == '"')
      return parseString();
    if (c == 't' || c == 'f')
      return parseBool();
    if (c == 'n')
      return parseNull();
    return parseNumber();
  }
  Json parseObject() {
    Json v;
    v.type = Json::Type::Object;
    i++; // {
    skipWs();
    if (i < s.size() && s[i] == '}') {
      i++;
      return v;
    }
    while (true) {
      skipWs();
      if (i >= s.size() || s[i] != '"')
        fail("expected string key");
      Json key = parseString();
      skipWs();
      if (i >= s.size() || s[i] != ':')
        fail("expected ':'");
      i++;
      Json val = parseValue();
      v.obj[key.str] = std::move(val);
      skipWs();
      if (i >= s.size())
        fail("unterminated object");
      if (s[i] == ',') {
        i++;
        continue;
      }
      if (s[i] == '}') {
        i++;
        break;
      }
      fail("expected ',' or '}'");
    }
    return v;
  }
  Json parseArray() {
    Json v;
    v.type = Json::Type::Array;
    i++; // [
    skipWs();
    if (i < s.size() && s[i] == ']') {
      i++;
      return v;
    }
    while (true) {
      Json val = parseValue();
      v.arr.push_back(std::move(val));
      skipWs();
      if (i >= s.size())
        fail("unterminated array");
      if (s[i] == ',') {
        i++;
        continue;
      }
      if (s[i] == ']') {
        i++;
        break;
      }
      fail("expected ',' or ']'");
    }
    return v;
  }
  Json parseString() {
    Json v;
    v.type = Json::Type::String;
    i++; // opening quote
    std::string out;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\' && i < s.size()) {
        char e = s[i++];
        switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        default: out += e; break;
        }
      } else {
        out += c;
      }
    }
    if (i >= s.size())
      fail("unterminated string");
    i++; // closing quote
    v.str = std::move(out);
    return v;
  }
  Json parseBool() {
    Json v;
    v.type = Json::Type::Bool;
    if (s.compare(i, 4, "true") == 0) {
      v.boolean = true;
      i += 4;
    } else if (s.compare(i, 5, "false") == 0) {
      v.boolean = false;
      i += 5;
    } else {
      fail("invalid literal");
    }
    return v;
  }
  Json parseNull() {
    Json v;
    v.type = Json::Type::Null;
    if (s.compare(i, 4, "null") == 0)
      i += 4;
    else
      fail("invalid literal");
    return v;
  }
  Json parseNumber() {
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
      i++;
    while (i < s.size() &&
           ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
            s[i] == 'E' || s[i] == '+' || s[i] == '-'))
      i++;
    Json v;
    v.type = Json::Type::Number;
    v.num = std::strtod(s.c_str() + start, nullptr);
    return v;
  }
};

// ---------------------------------------------------------------------------
// jithwdev.json-backed program config (replaces protobuf aicnwdesc).
// ---------------------------------------------------------------------------
struct IODescription {
  destination dest = DEST_DDR;
  uint32_t typeSize = 1; // bytes per element
  std::vector<int64_t> dims;
  uint64_t hostOffset = 0;
  uint64_t devOffset = 0;
  uint64_t baseAddrOffset = 0;
  uint8_t inSyncFence = 0;
  uint8_t outSyncFence = 0;
  bool allowPartial = false;
  bool noDoorbell = false;
  std::vector<uint32_t> nsps;
};

static uint32_t typeSizeFromName(const std::string &t) {
  if (t == "FloatTy")
    return 4;
  if (t == "Float16Ty")
    return 2;
  if (t == "Int8QTy")
    return 1;
  if (t == "UInt8QTy")
    return 1;
  if (t == "Int16QTy")
    return 2;
  if (t == "Int32QTy")
    return 4;
  if (t == "Int32ITy")
    return 4;
  if (t == "Int64ITy")
    return 8;
  if (t == "Int8Ty")
    return 1;
  fprintf(stderr, "mdgen: unknown data type '%s'\n", t.c_str());
  exit(1);
}

static destination destFromName(const std::string &d) {
  if (d == "DDR")
    return DEST_DDR;
  if (d == "L2TCM")
    return DEST_L2TCM;
  if (d == "VTCM")
    return DEST_VTCM;
  fprintf(stderr, "mdgen: unknown destination '%s'\n", d.c_str());
  exit(1);
}

static IODescription parseIO(const Json &j) {
  IODescription io;
  io.typeSize = typeSizeFromName(j.at("type").asStr("Int8Ty"));
  const Json &dims = j.at("dims");
  if (dims.type == Json::Type::Array) {
    for (auto &d : dims.arr)
      io.dims.push_back(d.asInt());
  } else if (dims.type == Json::Type::Number) {
    io.dims.push_back(dims.asInt());
  }
  io.hostOffset = j.at("hostOffset").asInt(0);
  io.devOffset = j.at("devOffset").asInt(0);
  io.baseAddrOffset = j.at("baseAddrOffset").asInt(0);
  io.dest = destFromName(j.at("dest").asStr("DDR"));
  io.inSyncFence = (uint8_t)j.at("inSyncFence").asInt(0);
  io.outSyncFence = (uint8_t)j.at("outSyncFence").asInt(0);
  io.allowPartial = j.at("allowPartial").asBool(false);
  io.noDoorbell = j.at("noDoorbell").asBool(false);
  const Json &nsps = j.at("nsps");
  if (nsps.type == Json::Type::Array)
    for (auto &n : nsps.arr)
      io.nsps.push_back((uint32_t)n.asInt());
  return io;
}

struct ProgramConfig {
  std::string name;
  uint32_t hwVersionMajor = 2;
  uint32_t hwVersionMinor = 0;
  uint32_t numThreads = 0;
  uint32_t numHMXThreads = 0;
  uint32_t numNSPs = 1;
  uint64_t heapSize = 0;
  bool qpcShim = false;
  bool singleVTCMPage = false;
  std::vector<IODescription> inputs;
  std::vector<IODescription> outputs;
  std::vector<IODescription> internalBuffers;
};

static ProgramConfig loadConfig(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "mdgen: cannot open %s\n", path.c_str());
    exit(1);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  Json root = JsonParser(ss.str()).parse();

  ProgramConfig c;
  c.name = root.at("name").asStr("");
  c.hwVersionMajor = (uint32_t)root.at("hwVersionMajor").asInt(2);
  c.hwVersionMinor = (uint32_t)root.at("hwVersionMinor").asInt(0);
  c.numThreads = (uint32_t)root.at("numThreads").asInt(0);
  c.numHMXThreads = (uint32_t)root.at("numHMXThreads").asInt(0);
  c.numNSPs = (uint32_t)root.at("numNSPs").asInt(1);
  c.heapSize = (uint64_t)root.at("heapSize").asInt(0);
  c.qpcShim = root.at("qpcShim").asBool(false);
  c.singleVTCMPage = root.at("singleVTCMPage").asBool(false);
  if (root.at("inputs").type == Json::Type::Array)
    for (auto &j : root.at("inputs").arr)
      c.inputs.push_back(parseIO(j));
  if (root.at("outputs").type == Json::Type::Array)
    for (auto &j : root.at("outputs").arr)
      c.outputs.push_back(parseIO(j));
  if (root.at("internalBuffers").type == Json::Type::Array)
    for (auto &j : root.at("internalBuffers").arr)
      c.internalBuffers.push_back(parseIO(j));
  return c;
}

static uint32_t getIOSize(const IODescription &io) {
  if (io.dims.empty()) {
    fprintf(stderr, "mdgen: Config Error: Buffers must have non-zero dims\n");
    exit(1);
  }
  int64_t dims = io.dims[0];
  for (size_t i = 1; i < io.dims.size(); i++)
    dims *= io.dims[i];
  if (dims == 0) {
    fprintf(stderr, "mdgen: Config Error: Buffers must have non-zero dims\n");
    exit(1);
  }
  return (uint32_t)(dims * io.typeSize);
}

// ---------------------------------------------------------------------------
// ProgramDesc -> constants.bin (mirrors lib/program/ProgramDesc.h).
// ---------------------------------------------------------------------------
class ProgramDesc {
public:
  ProgramDesc(uint16_t exitDB, uint16_t inputSem, uint16_t outputSem,
              uint32_t numThreads)
      : exitDB_(exitDB), numThreads_(numThreads), inputSem_(inputSem),
        outputSem_(outputSem) {}

  void addBuffer(const IODescription &desc, uint16_t waitDBNum,
                 uint16_t ioDBNum, uint32_t waitDBVal, uint32_t ioDBVal,
                 uint16_t ioMCID, uint16_t ioDBMCID, uint16_t buffMCID,
                 uint16_t nspMask, usageType_t usage, bool allowPartial) {
    memLoc_t location;
    switch (desc.dest) {
    case DEST_L2TCM: location = MEM_L2TCM; break;
    case DEST_VTCM: location = MEM_VTCM; break;
    case DEST_DDR: location = MEM_DDR; break;
    default:
      fprintf(stderr, "mdgen: invalid location\n");
      exit(1);
    }
    BufferDesc_t buffer{};
    buffer.location = location;
    buffer.offset = (uint32_t)(desc.devOffset + desc.baseAddrOffset);
    buffer.size = getIOSize(desc);
    buffer.waitDBNum = waitDBNum;
    buffer.ioDBNum = ioDBNum;
    buffer.waitDBVal = waitDBVal;
    buffer.ioDBVal = ioDBVal;
    buffer.ioMCID = ioMCID;
    buffer.ioDBMCID = ioDBMCID;
    buffer.buffMCID = buffMCID;
    buffer.nspMask = nspMask;
    buffer.usage = usage;
    buffer.allowPartial = allowPartial ? 1 : 0;
    buffers_.push_back(buffer);
    if (usage == USAGE_INPUT) {
      numInputBuffs_++;
      hasInputsMask_ |= nspMask;
    } else if (usage == USAGE_OUTPUT) {
      numOutputBuffs_++;
      hasOutputsMask_ |= nspMask;
    } else if (usage == USAGE_INTERNAL) {
      numInternalBuffs_++;
    }
  }

  void addUDMADescBuffer(const IODescription &udmaBuff,
                         uint32_t udmaDummyStartDescOffset, uint16_t nspMask) {
    udmaDummyStartDescOffset_ = udmaDummyStartDescOffset;
    udmaDescBuffNum_ = (uint32_t)buffers_.size();
    addBuffer(udmaBuff, 0, 0, 0, 0, 0, 0, 0, nspMask, USAGE_INTERNAL, false);
  }

  uint32_t serialize(std::ostream &f) {
    uint16_t numBuffs = (uint16_t)buffers_.size();
    uint32_t buffOffset = sizeof(SerializedProgramDesc_t);
    uint32_t size = buffOffset + (numBuffs * sizeof(BufferDesc_t));

    SerializedProgramDesc_t hdr{};
    hdr.serialVersion = SERIALIZED_PROGRAMDESC_VERSION;
    hdr.exitDB = exitDB_;
    hdr.size = size;
    hdr.numThreads = numThreads_;
    hdr.numBuffs = numBuffs;
    hdr.numInputBuffs = numInputBuffs_;
    hdr.numOutputBuffs = numOutputBuffs_;
    hdr.numInternalBuffs = numInternalBuffs_;
    hdr.inputSem = inputSem_;
    hdr.outputSem = outputSem_;
    hdr.hasInputsMask = hasInputsMask_;
    hdr.hasOutputsMask = hasOutputsMask_;
    hdr.buffersOffset = buffOffset;
    hdr.udmaDescBuffNum = udmaDescBuffNum_;
    hdr.udmaDummyStartDescOffset = udmaDummyStartDescOffset_;
    f.write((char *)&hdr, sizeof(hdr));
    for (auto &buff : buffers_)
      f.write((char *)&buff, sizeof(buff));
    return size;
  }

private:
  uint16_t exitDB_;
  uint32_t numThreads_{0};
  uint16_t numInputBuffs_{0};
  uint16_t numOutputBuffs_{0};
  uint16_t numInternalBuffs_{0};
  uint16_t inputSem_{0};
  uint16_t outputSem_{0};
  uint16_t hasInputsMask_{0};
  uint16_t hasOutputsMask_{0};
  uint32_t udmaDescBuffNum_{0};
  uint32_t udmaDummyStartDescOffset_{0};
  std::vector<BufferDesc_t> buffers_;
};

// ---------------------------------------------------------------------------
// Generator helpers (mirrors lib/program/Program.cpp).
// ---------------------------------------------------------------------------
static AICMDMulticastEntryAddrSpace getMCSpace(destination ioType) {
  if (ioType == DEST_L2TCM)
    return AICMDAddrSpaceL2TCM;
  if (ioType == DEST_VTCM)
    return AICMDAddrSpaceVTCM;
  fprintf(stderr, "mdgen: Invalid multicast destination\n");
  exit(1);
}

static AICMDDMAEntryAddrSpace getDMASpace(destination ioType) {
  return (ioType == DEST_DDR) ? AICMDDMAAddrSpaceDDR : AICMDDMAAddrSpaceMC;
}

static bool isMC(destination space) { return space != DEST_DDR; }

static uint32_t getNspMask(const IODescription &buff, uint32_t numNsps) {
  uint32_t allNspsMask = (0x1U << numNsps) - 1;
  if (buff.nsps.empty())
    return allNspsMask;
  uint32_t nspMask = 0;
  for (auto nspNum : buff.nsps) {
    if (nspNum >= numNsps) {
      fprintf(stderr, "mdgen: Config Error: Invalid nsp number %u\n", nspNum);
      exit(1);
    }
    nspMask |= (0x1U << nspNum);
  }
  return nspMask;
}

static void getNumThreads(uint32_t &numThreads, uint32_t &numHmxThreads,
                          uint32_t &numHvxThreads, const ProgramConfig &c) {
  numThreads = c.numThreads;
  numHmxThreads = c.numHMXThreads;
  if (numThreads == 0)
    numThreads = 4 + numHmxThreads; // Default to 4 HVX threads.
  if (numHmxThreads > numThreads) {
    fprintf(stderr, "mdgen: numHMXThreads > numThreads\n");
    exit(1);
  }
  numHvxThreads = numThreads - numHmxThreads;
  if ((int)numThreads > MAX_NUM_THREADS) {
    fprintf(stderr,
            "mdgen: Config Error: numThreads isn't valid. 1-6 supported\n");
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// Minimal ELF32 little-endian reader: find the value of the `_qaic_start`
// symbol in a linked Hexagon network.elf. Replaces the post-link
// `llvm-readelf -s | awk` step so the entrypoint feeds mdgen directly.
// ---------------------------------------------------------------------------
static uint64_t readQaicStartAddr(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "mdgen: cannot open ELF %s\n", path.c_str());
    exit(1);
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  if (buf.size() < 52 || buf[0] != 0x7f || buf[1] != 'E' || buf[2] != 'L' ||
      buf[3] != 'F') {
    fprintf(stderr, "mdgen: %s is not an ELF file\n", path.c_str());
    exit(1);
  }
  if (buf[4] != 1 /*ELFCLASS32*/ || buf[5] != 1 /*little-endian*/) {
    fprintf(stderr, "mdgen: %s is not a little-endian ELF32\n", path.c_str());
    exit(1);
  }
  auto rd16 = [&](size_t o) -> uint16_t {
    return (uint16_t)(buf[o] | (buf[o + 1] << 8));
  };
  auto rd32 = [&](size_t o) -> uint32_t {
    return (uint32_t)buf[o] | ((uint32_t)buf[o + 1] << 8) |
           ((uint32_t)buf[o + 2] << 16) | ((uint32_t)buf[o + 3] << 24);
  };
  uint32_t e_shoff = rd32(32);
  uint16_t e_shentsize = rd16(46);
  uint16_t e_shnum = rd16(48);
  uint16_t e_shstrndx = rd16(50);

  // ELF32 section header fields.
  auto shType = [&](uint16_t i) { return rd32(e_shoff + i * e_shentsize + 4); };
  auto shOffset = [&](uint16_t i) {
    return rd32(e_shoff + i * e_shentsize + 16);
  };
  auto shSize = [&](uint16_t i) {
    return rd32(e_shoff + i * e_shentsize + 20);
  };
  auto shLink = [&](uint16_t i) {
    return rd32(e_shoff + i * e_shentsize + 24);
  };
  auto shEntsize = [&](uint16_t i) {
    return rd32(e_shoff + i * e_shentsize + 36);
  };
  (void)e_shstrndx;

  const uint32_t SHT_SYMTAB = 2;
  const uint32_t SHT_DYNSYM = 11;
  for (uint16_t i = 0; i < e_shnum; i++) {
    uint32_t type = shType(i);
    if (type != SHT_SYMTAB && type != SHT_DYNSYM)
      continue;
    uint32_t symoff = shOffset(i);
    uint32_t symsize = shSize(i);
    uint32_t entsize = shEntsize(i);
    uint32_t strtab = shLink(i);
    uint32_t stroff = shOffset((uint16_t)strtab);
    uint32_t strsize = shSize((uint16_t)strtab);
    if (entsize == 0)
      entsize = 16; // ELF32 Sym size
    for (uint32_t off = symoff; off + entsize <= symoff + symsize;
         off += entsize) {
      uint32_t st_name = rd32(off);
      uint32_t st_value = rd32(off + 4);
      if (st_name == 0 || stroff + st_name >= stroff + strsize)
        continue;
      const char *nm = (const char *)&buf[stroff + st_name];
      if (std::strcmp(nm, "_qaic_start") == 0)
        return st_value;
    }
  }
  fprintf(stderr, "mdgen: _qaic_start symbol not found in %s\n", path.c_str());
  exit(1);
}
} // namespace

int main(int argc, char **argv) {
  const char *jsonPath = nullptr;
  uint64_t entryPoint = 0;
  bool haveEntryPoint = false;
  const char *elfPath = nullptr;
  const char *metadataOut = "metadata.bin";
  const char *constantsOut = "constants.bin";
  const char *constantsDescOut = "constantsdesc.bin";
  const char *mdStructOut = nullptr;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-o" && i + 1 < argc)
      metadataOut = argv[++i];
    else if (a == "--constants" && i + 1 < argc)
      constantsOut = argv[++i];
    else if (a == "--constantsdesc" && i + 1 < argc)
      constantsDescOut = argv[++i];
    else if (a == "--md-struct" && i + 1 < argc)
      mdStructOut = argv[++i];
    else if (a == "--elf" && i + 1 < argc)
      elfPath = argv[++i];
    else if (a == "-e" || a == "--entrypoint") {
      if (i + 1 < argc) {
        entryPoint = std::strtoull(argv[++i], nullptr, 0);
        haveEntryPoint = true;
      }
    } else if (!jsonPath)
      jsonPath = argv[i];
    else {
      entryPoint = std::strtoull(argv[i], nullptr, 0);
      haveEntryPoint = true;
    }
  }
  if (!jsonPath) {
    fprintf(stderr,
            "usage: mdgen <jithwdev.json> (<entrypoint> | --elf network.elf) "
            "[-o metadata.bin] [--constants constants.bin] "
            "[--constantsdesc constantsdesc.bin] [--md-struct md_struct.bin]\n");
    return 1;
  }
  if (elfPath) {
    // Prefer the entrypoint read directly from the linked ELF.
    entryPoint = readQaicStartAddr(elfPath);
    haveEntryPoint = true;
  }
  if (!haveEntryPoint) {
    fprintf(stderr, "mdgen: no entrypoint given (pass <entrypoint> or --elf)\n");
    return 1;
  }

  ProgramConfig c = loadConfig(jsonPath);

  if (c.hwVersionMajor != 2) {
    fprintf(stderr, "mdgen: Config Error: hwVersionMajor must be 2\n");
    return 1;
  }
  if (c.hwVersionMinor != 0) {
    fprintf(stderr, "mdgen: Config Error: hwVersionMinor must be 0\n");
    return 1;
  }

  MetadataFlatbufferWriter metadata(c.hwVersionMajor, c.hwVersionMinor);

  uint16_t numNsps = (uint16_t)c.numNSPs;
  if (numNsps < 1 || numNsps > MAX_NUM_CORES) {
    fprintf(stderr, "mdgen: Config Error: numNSPs must be 1-16\n");
    return 1;
  }
  uint32_t allNspsMask = (0x1U << numNsps) - 1;

  metadata.setNetworkName(c.name);
  metadata.setNumNSPs(numNsps);
  metadata.setNumSemaphores(COMPUTE_MAX_SEMAPHORES);

  uint16_t fileNum = 0;
  uint64_t hostOffset = 0;
  uint8_t dynamic_entry = 0;
  const uint16_t inputSem = 0;
  const uint16_t outputSem = 1;

  size_t ddrBuffersSize = c.qpcShim ? 0x2000000 : 0;
  size_t l2tcmBuffersSize = 1;
  size_t vtcmBuffersSize = 1;
  uint16_t input_port_id = 100;
  uint16_t output_port_id = 101;

  uint32_t numThreads, numHmxThreads, numHvxThreads;
  constexpr uint32_t MAX_HVX_THREADS = 4;
  getNumThreads(numThreads, numHmxThreads, numHvxThreads, c);
  numHvxThreads = std::min(numHvxThreads, MAX_HVX_THREADS);

  uint16_t mcId = 0;
  uint64_t DBNum = 0;
  uint32_t DBData = 1;
  int numDBs = 1; // Always have exit DB
  for (auto const &buff : c.inputs)
    if (!buff.noDoorbell)
      numDBs++;
  for (auto const &buff : c.outputs)
    if (!buff.noDoorbell)
      numDBs++;
  for (auto const &buff : c.internalBuffers)
    if (!buff.noDoorbell)
      numDBs++;
  const uint32_t MAX_DBS = c.qpcShim ? 32768 : 281;
  if (numDBs < (int)MAX_DBS)
    numDBs = MAX_DBS;

  const unsigned int DBSpaceSize = (numDBs * DB_SIZE);
  metadata.addHostMulticastEntry(allNspsMask, DBSpaceSize);
  for (uint16_t core = 0; core < numNsps; ++core)
    metadata.addNSPMulticastEntry(core, dynamic_entry,
                                  allNspsMask & ~(0x1U << core), DBSpaceSize,
                                  AICMDAddrSpaceL2TCM, 0);

  uint32_t udmaDummyStartDescOffset;
  uint32_t udmaBufferStartOffset;
  if (c.qpcShim) {
    uint32_t reservedCommandSetSize = 0x20000;
    udmaDummyStartDescOffset =
        alignTo(DBSpaceSize + reservedCommandSetSize, CACHE_LINE_SIZE);
    udmaBufferStartOffset = alignTo(
        udmaDummyStartDescOffset + DMA_DESCRIPTOR_SIZE, CACHE_LINE_SIZE);
  } else {
    udmaDummyStartDescOffset = alignTo(DBSpaceSize, CACHE_LINE_SIZE);
    udmaBufferStartOffset = alignTo(
        udmaDummyStartDescOffset + DMA_DESCRIPTOR_SIZE, CACHE_LINE_SIZE);
  }
  uint32_t udmaBufferSize =
      numThreads * CACHE_LINE_SIZE * NUM_UDMA_CACHELINES_PER_THREAD;

  metadata.initL2TCMResize(udmaDummyStartDescOffset + DMA_DESCRIPTOR_SIZE);

  size_t reservedL2TCMSize = udmaDummyStartDescOffset + udmaBufferSize;
  l2tcmBuffersSize = std::max(l2tcmBuffersSize, reservedL2TCMSize);

  // Initialize the dummy descriptor to done. DMADescriptor is four uint32_t
  // words; only the "done" bit (word1 bit 31) is set.
  uint32_t dummyDesc[DMA_DESCRIPTOR_SIZE / sizeof(uint32_t)] = {0, 0, 0, 0};
  dummyDesc[1] = (1u << 31); // done bit
  for (uint64_t i = 0; i < DMA_DESCRIPTOR_SIZE / sizeof(uint32_t); i++)
    metadata.initL2TCMWord(udmaDummyStartDescOffset + i * sizeof(uint32_t),
                           dummyDesc[i]);

  metadata.addPort(input_port_id, AicMetadataFlat::AICMDPortType_AICMDPortUserIO);
  metadata.addPort(output_port_id,
                   AicMetadataFlat::AICMDPortType_AICMDPortUserIO);

  metadata.setExitDoorbellOffset(DBSpaceSize - DB_SIZE);
  metadata.initL2TCMWord(DBSpaceSize - DB_SIZE, 0x0);

  mcId++;

  ProgramDesc progDesc(/*exitDB*/ numDBs - 1, inputSem, outputSem, numThreads);

  // Collected DMA requests for md_struct.bin (parallel to metadata.addDMARequest).
  std::vector<MdDmaRequest> mdDmaRequests;

  auto processBuff = [&](const IODescription &buff, usageType_t usage,
                         uint16_t semNum, uint16_t semWaitVal,
                         uint16_t semInitVal, bool last) {
    MetadataFlatbufferWriter::DynamicSharedDDRStatus unused;
    (void)unused;
    SemaphoreOps semaphoreOps;
    DoorbellOps doorbellOps;

    uint32_t dmaSize = getIOSize(buff);
    destination memType = buff.dest;
    uint64_t devOffset = buff.devOffset;
    uint64_t baseAddrOffset = buff.baseAddrOffset;
    uint32_t nspMask = getNspMask(buff, numNsps);

    if (isMC(memType)) {
      if ((baseAddrOffset & ((1U << 12) - 1)) != 0) {
        fprintf(stderr, "mdgen: Config Error: baseAddrOffset must be "
                        "multiples of 4096\n");
        exit(1);
      }
      if (usage == USAGE_OUTPUT && numNsps != 1 &&
          ((nspMask & (nspMask - 1)) != 0)) {
        fprintf(stderr, "mdgen: Config Error: Each output buffer in "
                        "L2TCM/VTCM must be limited to a single NSP\n");
        exit(1);
      }
      if (usage != USAGE_INTERNAL)
        metadata.addHostMulticastEntry(nspMask, dmaSize);
      for (uint16_t core = 0; core < numNsps; ++core)
        metadata.addNSPMulticastEntry(
            core, dynamic_entry, (usage == USAGE_INTERNAL) ? nspMask : 0,
            dmaSize, getMCSpace(memType), baseAddrOffset);
      mcId++;
      if (memType == DEST_L2TCM) {
        l2tcmBuffersSize =
            std::max(l2tcmBuffersSize, baseAddrOffset + devOffset + dmaSize);
        if (l2tcmBuffersSize > L2TCM_MAX_SIZE) {
          fprintf(stderr, "mdgen: Config Error: L2TCM buffers can't go past "
                          "%u\n",
                  L2TCM_MAX_SIZE);
          exit(1);
        }
        if (!c.qpcShim && (devOffset + baseAddrOffset < reservedL2TCMSize)) {
          fprintf(stderr, "mdgen: Config Error: L2TCM buffers can't start "
                          "before offset %zu\n",
                  reservedL2TCMSize);
          exit(1);
        }
      } else {
        vtcmBuffersSize =
            std::max(vtcmBuffersSize, baseAddrOffset + devOffset + dmaSize);
        if (vtcmBuffersSize > VTCM_MAX_SIZE) {
          fprintf(stderr, "mdgen: Config Error: VTCM buffers can't go past "
                          "%u\n",
                  VTCM_MAX_SIZE);
          exit(1);
        }
      }
    } else {
      if (baseAddrOffset != 0) {
        fprintf(stderr, "mdgen: Config Error: DDR buffers don't support "
                        "baseAddrOffset\n");
        exit(1);
      }
      ddrBuffersSize = std::max(ddrBuffersSize, devOffset + dmaSize);
    }

    if (usage != USAGE_INTERNAL) {
      metadata.addSemaphoreOp(semaphoreOps, AICMDSemaphoreCmdWAITEQ, semNum,
                              semWaitVal, AICMDSemaphoreSyncPre,
                              buff.inSyncFence, buff.outSyncFence);
      if (last)
        metadata.addSemaphoreOp(semaphoreOps, AICMDSemaphoreCmdINIT, semNum,
                                semInitVal, AICMDSemaphoreSyncPost,
                                buff.inSyncFence, buff.outSyncFence);
    }

    uint64_t DBOffset = DBNum * DB_SIZE;
    if (usage != USAGE_INTERNAL)
      metadata.addDoorbellOp(doorbellOps, AICMDDoorballOpSize32, /*mcId*/ 0,
                             DBOffset, DBData);

    if (usage == USAGE_INPUT) {
      metadata.initL2TCMWord(DBOffset, 0);
    } else if (buff.allowPartial) {
      fprintf(stderr, "mdgen: Only Input buffers can have allowPartial set\n");
      exit(1);
    } else if (usage == USAGE_OUTPUT) {
      metadata.initL2TCMWord(DBOffset, c.qpcShim ? 0 : DBData);
    } else if (usage == USAGE_INTERNAL) {
      metadata.initL2TCMWord(DBOffset, 0);
    }

    progDesc.addBuffer(buff, /*waitDBNum*/ DBNum, /*ioDBNum*/ 0,
                       /*waitDBVal*/ DBData, /*ioDBVal*/ DBData, /*ioMCID*/ 0,
                       /*ioDBMCID*/ 0,
                       /*buffMCID*/ isMC(memType) ? mcId - 1 : 0,
                       /*nspMask*/ (uint16_t)nspMask, /*usage*/ usage,
                       /*allowPartial*/ buff.allowPartial);
    DBNum++;

    if (usage != USAGE_INTERNAL) {
      AICMDDMADirection dir = (usage == USAGE_INPUT) ? AICMDDMAIn : AICMDDMAOut;
      uint16_t reqMcId = (isMC(memType) ? mcId - 1 : 0);
      uint16_t reqPort = (usage == USAGE_INPUT) ? input_port_id : output_port_id;
      metadata.addDMARequest(fileNum, hostOffset, getDMASpace(memType),
                             devOffset, dmaSize, dir, reqPort, reqMcId,
                             semaphoreOps, doorbellOps,
                             /*transactionId*/
                             AicMetadataFlat::AICMDDMAReserved_AICMDDMATransactionIdNone);

      // Capture the same request for md_struct.bin.
      MdDmaRequest md{};
      md.dev_addr_space = getDMASpace(memType);
      md.num = fileNum;
      md.dev_addr_offset = devOffset;
      md.host_addr_offset = hostOffset;
      md.size = dmaSize;
      md.in_out = dir;
      md.mc_id = reqMcId;
      md.port_id = reqPort;
      md.semaphore_count = (uint32_t)semaphoreOps.size();
      if (!doorbellOps.empty()) {
        md.doorbell.data = doorbellOps[0].data;
        md.doorbell.mc_id = doorbellOps[0].mcId;
        md.doorbell.offset = doorbellOps[0].offset;
        md.doorbell.size = doorbellOps[0].size;
      }
      for (size_t s = 0; s < semaphoreOps.size() && s < 4; s++) {
        md.semaphore_ops[s].in_sync_fence = semaphoreOps[s].inSyncFence;
        md.semaphore_ops[s].out_sync_fence = semaphoreOps[s].outSyncFence;
        md.semaphore_ops[s].pre_or_post = semaphoreOps[s].preOrPost;
        md.semaphore_ops[s].sem_num = semaphoreOps[s].semNum;
        md.semaphore_ops[s].sem_op = semaphoreOps[s].semOp;
        md.semaphore_ops[s].sem_value = semaphoreOps[s].semValue;
      }
      mdDmaRequests.push_back(md);

      fileNum++;
    }
  };

  uint16_t inMask = 0;
  uint16_t outMask = 0;
  for (auto const &buff : c.inputs)
    inMask |= getNspMask(buff, numNsps);
  for (auto const &buff : c.outputs)
    outMask |= getNspMask(buff, numNsps);

  // Input buffers
  uint16_t semNum = inputSem;
  uint16_t semInitVal = 0;
  uint16_t semWaitVal = std::bitset<MAX_NUM_CORES>(inMask).count();
  int lastIndex = (int)c.inputs.size() - 1;
  metadata.initSemaphore(semNum, semInitVal);
  for (int i = 0; i < (int)c.inputs.size(); i++)
    processBuff(c.inputs[i], USAGE_INPUT, semNum, semWaitVal, semInitVal,
                i == lastIndex);

  // Output buffers
  semNum = outputSem;
  semInitVal =
      c.qpcShim ? 0 : std::bitset<MAX_NUM_CORES>(outMask).count();
  semWaitVal =
      c.qpcShim ? std::bitset<MAX_NUM_CORES>(outMask).count() : 0;
  lastIndex = (int)c.outputs.size() - 1;
  metadata.initSemaphore(semNum, semInitVal);
  for (int i = 0; i < (int)c.outputs.size(); i++)
    processBuff(c.outputs[i], USAGE_OUTPUT, semNum, semWaitVal, semInitVal,
                i == lastIndex);

  // Internal buffers
  for (auto const &buff : c.internalBuffers)
    processBuff(buff, USAGE_INTERNAL, 0, 0, 0, false);

  // UDMA descriptors buffer
  IODescription udmaBuff;
  udmaBuff.typeSize = 1; // Int8Ty
  udmaBuff.dims.push_back(udmaBufferSize);
  udmaBuff.hostOffset = 0;
  udmaBuff.devOffset = udmaBufferStartOffset;
  udmaBuff.baseAddrOffset = 0;
  udmaBuff.dest = DEST_L2TCM;
  progDesc.addUDMADescBuffer(udmaBuff, udmaDummyStartDescOffset, allNspsMask);

  metadata.setVTCMSize(vtcmBuffersSize);
  metadata.setL2TCMSize(l2tcmBuffersSize);
  metadata.setStaticSharedDDRSize(ddrBuffersSize);
  metadata.setStaticSharedDDRECCEnabled(false);
  metadata.setDynamicSharedDDRSize(0);
  metadata.setDynamicSharedDDRECCEnabled(false);
  metadata.setNetworkHeapSize(c.heapSize);
  metadata.setSingleVTCMPage(c.singleVTCMPage);

  // constants.bin
  std::ofstream constantsFile(constantsOut,
                              std::ios::binary | std::ios::out);
  if (!constantsFile) {
    fprintf(stderr, "mdgen: Unable to create %s\n", constantsOut);
    return 1;
  }
  uint32_t progDescSize = progDesc.serialize(constantsFile);
  constantsFile.close();

  metadata.setStaticConstantsSize(progDescSize);
  metadata.setStaticConstantsECCEnabled(false);
  metadata.setDynamicConstantsSize(0);
  metadata.setDynamicConstantsECCEnabled(false);
  metadata.addConstantMapping(allNspsMask, /*offset*/ 0, progDescSize);

  // constantsdesc.bin
  std::ofstream constDescFile(constantsDescOut,
                              std::ios::binary | std::ios::out);
  if (!constDescFile) {
    fprintf(stderr, "mdgen: Unable to create %s\n", constantsDescOut);
    return 1;
  }
  AICConstantDescriptor constDesc = {};
  constDesc.staticConstantsSize = metadata.getStaticConstantsSize();
  constDesc.dynamicConstantsSize = metadata.getDynamicConstantsSize();
  constDesc.staticConstantsECCEnabled = metadata.getStaticConstantsECCEnabled();
  constDesc.dynamicConstantsECCEnabled =
      metadata.getDynamicConstantsECCEnabled();
  constDescFile.write((char *)&constDesc, sizeof(constDesc));
  constDescFile.close();

  // Thread descriptors.
  nnc_activate_fp entryPointFn = reinterpret_cast<nnc_activate_fp>(entryPoint);
  for (unsigned int i = 0; i < numHvxThreads; i++)
    metadata.addThreadDescriptor(entryPointFn, false, true);
  for (unsigned int i = 0; i < numHmxThreads; i++)
    metadata.addThreadDescriptor(entryPointFn, true, false);
  for (unsigned int i = numHvxThreads + numHmxThreads; i < numThreads; i++)
    metadata.addThreadDescriptor(entryPointFn, false, false);

  metadata.finalize();

  if (!metadata.writeMetadata(metadataOut)) {
    fprintf(stderr, "mdgen: failed to write %s\n", metadataOut);
    return 1;
  }

  // md_struct.bin: compact metadata the driver reads at load time.
  if (mdStructOut) {
    std::ofstream mdFile(mdStructOut, std::ios::binary | std::ios::out);
    if (!mdFile) {
      fprintf(stderr, "mdgen: Unable to create %s\n", mdStructOut);
      return 1;
    }
    MdHeader hdr{};
    std::memset(hdr.network_name, 0, sizeof(hdr.network_name));
    std::strncpy(hdr.network_name, c.name.c_str(),
                 sizeof(hdr.network_name) - 1);
    hdr.nsp_count = numNsps;
    hdr.semaphores_count = COMPUTE_MAX_SEMAPHORES;
    hdr.l2tcm_size = (uint32_t)l2tcmBuffersSize;
    hdr.vtcm_size = (uint32_t)vtcmBuffersSize;
    hdr.dma_req_count = (uint32_t)mdDmaRequests.size();
    // constantMappings always present (addConstantMapping called above).
    hdr.static_constants_exist = 1;
    mdFile.write((char *)&hdr, sizeof(hdr));
    for (auto &req : mdDmaRequests)
      mdFile.write((char *)&req, sizeof(req));
    mdFile.close();
  }

  return 0;
}
