const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;

const MAX_METADATA_PAIRS: usize = 16;
const MAX_KEY_LEN: usize = 64;
const MAX_STRING_LEN: usize = 256;
const MAX_METADATA_BYTES: usize = 1024;

const Value = union(enum) {
    boolean: bool,
    integer: i64,
    string: []const u8,
};
const MetadataPair = struct {
    key: []const u8,
    value: Value,
};
const Condition = union(enum) {
    has: []const u8,
    eq: struct { key: []const u8, value: Value },
};

fn value_eq(a: Value, b: Value) bool {
    return switch (a) {
        .boolean => |av| switch (b) {
            .boolean => |bv| av == bv,
            else => false,
        },
        .integer => |av| switch (b) {
            .integer => |bv| av == bv,
            else => false,
        },
        .string => |av| switch (b) {
            .string => |bv| std.mem.eql(u8, av, bv),
            else => false,
        },
    };
}

// simple linear scan
fn find_value(metadata: []const MetadataPair, key: []const u8) ?Value {
    for (metadata) |pair| {
        if (std.mem.eql(u8, pair.key, key)) return pair.value;
    }
    return null;
}

fn validate_metadata(pairs: []const MetadataPair) !void {
    if (pairs.len > MAX_METADATA_PAIRS) return error.MetadataTooManyPairs;

    var total_bytes: usize = 0;

    for (pairs, 0..) |pair, i| {
        if (pair.key.len == 0) return error.MetadataEmptyKey;
        if (pair.key.len > MAX_KEY_LEN) return error.MetadataKeyTooLong;

        total_bytes += pair.key.len;

        switch (pair.value) {
            .string => |s| {
                if (s.len > MAX_STRING_LEN) return error.MetadataStringTooLong;
                total_bytes += s.len;
            },
            else => {},
        }

        for (pairs[0..i]) |prev| {
            if (std.mem.eql(u8, prev.key, pair.key)) return error.MetadataDuplicateKey;
        }
    }

    if (total_bytes > MAX_METADATA_BYTES) return error.MetadataTooLarge;
}

fn matches_filter(metadata: []const MetadataPair, filter: []const Condition) bool {
    for (filter) |cond| {
        switch (cond) {
            .has => |key| {
                if (find_value(metadata, key) == null) return false;
            },
            .eq => |eq_cond| {
                const actual = find_value(metadata, eq_cond.key) orelse return false;
                if (!value_eq(actual, eq_cond.value)) return false;
            },
        }
    }
    return true;
}

const Header = extern struct {
    magic: u32,
    version: u32,
    dim: u32,
    _padding: u32,
    count: u64,
    offset_table_pos: u64,
    blob_pos: u64,

    comptime {
        assert(@sizeOf(Header) == 40);
        assert(@alignOf(Header) == 8);
    }
};

const OffsetEntry = extern struct {
    offset: u64,
    length: u32,
    _padding: u32,

    comptime {
        assert(@sizeOf(OffsetEntry) == 16);
        assert(@alignOf(OffsetEntry) == 8);
    }
};

fn serialize_metadata(writer: *std.io.Writer, metadata: []const MetadataPair) !void {
    try writer.writeInt(u16, @intCast(metadata.len), .little);

    for (metadata) |pair| {
        try writer.writeInt(u8, @intCast(pair.key.len), .little);
        try writer.writeAll(pair.key);

        // value_type: 0 => boolean, 1 => integer, 2 => string
        switch (pair.value) {
            .boolean => |b| {
                try writer.writeInt(u8, 0, .little);
                try writer.writeInt(u8, @intFromBool(b), .little);
            },
            .integer => |i| {
                try writer.writeInt(u8, 1, .little);
                try writer.writeInt(i64, i, .little);
            },
            .string => |s| {
                try writer.writeInt(u8, 2, .little);
                try writer.writeInt(u16, @intCast(s.len), .little);
                try writer.writeAll(s);
            },
        }
    }
}

fn deserialize_metadata(reader: *std.io.Reader, allocator: std.mem.Allocator) ![]const MetadataPair {
    const pair_count = try reader.takeInt(u16, .little);

    const pairs = try allocator.alloc(MetadataPair, pair_count);

    for (pairs) |*pair| {
        const key_len = try reader.takeInt(u8, .little);
        const key = try allocator.alloc(u8, key_len);
        try reader.readSliceAll(key);
        pair.key = key;

        const value_type = try reader.takeInt(u8, .little);
        pair.value = switch (value_type) {
            0 => .{ .boolean = try reader.takeInt(u8, .little) != 0 },
            1 => .{ .integer = try reader.takeInt(i64, .little) },
            2 => blk: {
                const str_len = try reader.takeInt(u16, .little);
                const str = try allocator.alloc(u8, str_len);
                try reader.readSliceAll(str);
                break :blk .{ .string = str };
            },
            else => return error.InvalidMetadataValueType,
        };
    }

    return pairs;
}

// "ELQ" + 0x01 (in hex) = 0x00514C45 (Depending on endianness, but let's just use a constant)
const MAGIC_NUMBER: u32 = 0x514C4501;

pub const DistanceMetric = enum {
    Cosine, // Normalize vectors and use dot product
    DotProduct, // Raw dot product without normalization
    Euclidean, // Use Euclidean distance
};

fn euclidean_distance(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    const diff = a - b;
    return @sqrt(@reduce(.Add, diff * diff));
}

fn euclidean_score(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    return -euclidean_distance(dim, a, b);
}

fn dot(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    return @reduce(.Add, a * b);
}

fn norm(comptime dim: usize, v: @Vector(dim, f32)) f32 {
    return @sqrt(@reduce(.Add, v * v));
}

fn normalize(comptime dim: usize, v: @Vector(dim, f32)) @Vector(dim, f32) {
    const n = norm(dim, v);
    if (n == 0.0) return v;
    return v / @as(@Vector(dim, f32), @splat(n));
}

pub fn VectorDB(comptime dim: usize, comptime metric: DistanceMetric) type {
    return struct {
        ids: std.ArrayList(u64),
        vectors: std.ArrayList(@Vector(dim, f32)),
        metadatas: std.ArrayList([]const MetadataPair),
        allocator: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        next_id: u64,

        const Self = @This();
        const Vec = @Vector(dim, f32);

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .ids = .{}, .vectors = .{}, .metadatas = .{}, .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator), .next_id = 1 };
        }

        pub fn deinit(self: *Self) void {
            self.ids.deinit(self.allocator);
            self.vectors.deinit(self.allocator);
            self.metadatas.deinit(self.allocator);

            self.arena.deinit();
        }

        pub fn add(self: *Self, raw_vec: Vec, metadata: []const MetadataPair) !u64 {
            try validate_metadata(metadata);

            const vec = if (metric == .Cosine) normalize(dim, raw_vec) else raw_vec;

            const id = self.next_id;
            if (id == std.math.maxInt(u64)) return error.MaxIdReached;

            const arena_allocator = self.arena.allocator();

            var copied_pairs = try arena_allocator.alloc(MetadataPair, metadata.len);
            for (metadata, 0..) |pair, i| {
                const key_copy = try arena_allocator.dupe(u8, pair.key);

                const value_copy: Value = switch (pair.value) {
                    .boolean => |b| .{ .boolean = b },
                    .integer => |n| .{ .integer = n },
                    .string => |s| .{ .string = try arena_allocator.dupe(u8, s) },
                };

                copied_pairs[i] = .{ .key = key_copy, .value = value_copy };
            }

            try self.vectors.append(self.allocator, vec);
            errdefer _ = self.vectors.pop();

            try self.ids.append(self.allocator, id);
            errdefer _ = self.ids.pop();

            try self.metadatas.append(self.allocator, copied_pairs);
            errdefer _ = self.metadatas.pop();

            self.next_id += 1;
            return id;
        }

        pub const Result = struct {
            id: u64,
            score: f32,
        };

        fn compare_score(_: void, a: Result, b: Result) std.math.Order {
            return std.math.order(a.score, b.score);
        }

        pub fn search(self: *Self, raw_query: Vec, k: usize) ![]Result {
            if (self.vectors.items.len == 0 or k == 0) return try self.allocator.alloc(Result, 0);

            const query = if (metric == .Cosine) normalize(dim, raw_query) else raw_query;
            const PriorityQueue = std.PriorityDequeue(Result, void, compare_score);

            var priority_queue = PriorityQueue.init(self.allocator, {});
            defer priority_queue.deinit();

            for (self.ids.items, self.vectors.items) |id, vec| {
                // Normally we can do cosine similarity to find similarity:
                // cosine(a, b) = dot(a, b) / (||a|| * ||b||)
                // if we normalize all vectors to unit length, we can just do:
                // cosine(a, b) = dot(a, b)  ← much faster, no sqrt or division per pair
                const score = switch (metric) {
                    .Cosine, .DotProduct => dot(dim, query, vec),
                    .Euclidean => euclidean_score(dim, query, vec),
                };

                const entry = Result{ .id = id, .score = score };

                if (priority_queue.len < k) {
                    try priority_queue.add(entry);
                } else if (score > priority_queue.peekMin().?.score) {
                    _ = priority_queue.removeMin();
                    try priority_queue.add(entry);
                }
            }

            var i = priority_queue.len;
            var results = try self.allocator.alloc(Result, i);
            while (priority_queue.removeMinOrNull()) |res| {
                i -= 1;
                results[i] = res;
            }

            return results;
        }

        pub fn search_filtered(self: *Self, raw_query: Vec, k: usize, filter: []const Condition) ![]Result {
            if (self.vectors.items.len == 0 or k == 0) return try self.allocator.alloc(Result, 0);

            std.debug.assert(self.ids.items.len == self.vectors.items.len);
            std.debug.assert(self.metadatas.items.len == self.vectors.items.len);

            const query = if (metric == .Cosine) normalize(dim, raw_query) else raw_query;
            const PriorityQueue = std.PriorityDequeue(Result, void, compare_score);

            var priority_queue = PriorityQueue.init(self.allocator, {});
            defer priority_queue.deinit();

            for (self.ids.items, self.vectors.items, self.metadatas.items) |id, vec, metadata| {
                if (!matches_filter(metadata, filter)) continue;

                // Normally we can do cosine similarity to find similarity:
                // cosine(a, b) = dot(a, b) / (||a|| * ||b||)
                // if we normalize all vectors to unit length, we can just do:
                // cosine(a, b) = dot(a, b)  ← much faster, no sqrt or division per pair
                const score = switch (metric) {
                    .Cosine, .DotProduct => dot(dim, query, vec),
                    .Euclidean => euclidean_score(dim, query, vec),
                };

                const entry = Result{ .id = id, .score = score };

                if (priority_queue.len < k) {
                    try priority_queue.add(entry);
                } else if (score > priority_queue.peekMin().?.score) {
                    _ = priority_queue.removeMin();
                    try priority_queue.add(entry);
                }
            }

            var i = priority_queue.len;
            var results = try self.allocator.alloc(Result, i);
            while (priority_queue.removeMinOrNull()) |res| {
                i -= 1;
                results[i] = res;
            }

            return results;
        }

        pub fn save(self: *Self, path: []const u8) !void {
            var file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            const count = self.vectors.items.len;

            var allocating = std.io.Writer.Allocating.init(self.allocator);
            defer allocating.deinit();

            var offsets = try self.allocator.alloc(OffsetEntry, count);
            defer self.allocator.free(offsets);

            for (self.metadatas.items, 0..) |meta, i| {
                const start_offset = allocating.written().len;

                try serialize_metadata(&allocating.writer, meta);

                offsets[i] = OffsetEntry{
                    .offset = @intCast(start_offset),
                    .length = @intCast(allocating.written().len - start_offset),
                    ._padding = 0,
                };
            }

            const header_size: u64 = @sizeOf(Header);
            const vectors_size: u64 = count * dim * @sizeOf(f32);
            const ids_size: u64 = count * @sizeOf(u64);
            const offset_table_pos = header_size + vectors_size + ids_size;
            const blob_pos = offset_table_pos + (count * @sizeOf(OffsetEntry));

            const header = Header{
                .magic = MAGIC_NUMBER,
                .version = 1,
                .dim = @intCast(dim),
                ._padding = 0,
                .count = count,
                .offset_table_pos = offset_table_pos,
                .blob_pos = blob_pos,
            };

            try file.writeAll(std.mem.asBytes(&header));
            try file.writeAll(std.mem.sliceAsBytes(self.vectors.items));
            try file.writeAll(std.mem.sliceAsBytes(self.ids.items));
            try file.writeAll(std.mem.sliceAsBytes(offsets));
            try file.writeAll(allocating.written());
        }

        pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
            var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
            defer file.close();

            var header_bytes: [@sizeOf(Header)]u8 = undefined;
            const header_read = try file.readAll(&header_bytes);
            if (header_read != @sizeOf(Header)) {
                return error.UnexpectedEndOfFile;
            }

            const header: *const Header = @ptrCast(@alignCast(&header_bytes));
            if (header.magic != MAGIC_NUMBER) {
                return error.InvalidFileFormat;
            }
            if (header.dim != dim) {
                return error.DimensionMismatch;
            }

            const count: usize = @intCast(header.count);

            var db = Self.init(allocator);
            errdefer db.deinit();

            try db.vectors.resize(allocator, count);
            const vector_bytes = std.mem.sliceAsBytes(db.vectors.items);
            const vector_bytes_read = try file.readAll(vector_bytes);
            if (vector_bytes_read != vector_bytes.len) return error.UnexpectedEndOfFile;

            try db.ids.resize(allocator, count);
            const id_bytes = std.mem.sliceAsBytes(db.ids.items);
            const id_bytes_read = try file.readAll(id_bytes);
            if (id_bytes_read != id_bytes.len) return error.UnexpectedEndOfFile;

            const offsets = try allocator.alloc(OffsetEntry, count);
            defer allocator.free(offsets);

            const offset_bytes = std.mem.sliceAsBytes(offsets);
            const offset_bytes_read = try file.readAll(offset_bytes);
            if (offset_bytes_read != offset_bytes.len) return error.UnexpectedEndOfFile;

            try db.metadatas.resize(allocator, count);

            for (offsets, 0..) |entry, i| {
                if (entry.length == 0) {
                    db.metadatas.items[i] = &.{};
                    continue;
                }

                try file.seekTo(header.blob_pos + entry.offset);

                // Read the raw bytes for this metadata entry
                const meta_bytes = try allocator.alloc(u8, entry.length);
                defer allocator.free(meta_bytes);
                const meta_read = try file.readAll(meta_bytes);
                if (meta_read != entry.length) return error.UnexpectedEndOfFile;

                // Create a Reader from the byte slice
                var reader = std.io.Reader.fixed(meta_bytes);
                db.metadatas.items[i] = try deserialize_metadata(&reader, db.arena.allocator());
            }

            // try db.metadatas.resize(allocator, count);
            // const empty_metadata: []const MetadataPair = &.{};
            // for (db.metadatas.items) |*m| m.* = empty_metadata;

            var max_id: u64 = 0;
            for (db.ids.items) |id| max_id = @max(max_id, id);
            db.next_id = if (max_id == std.math.maxInt(u64)) max_id else max_id + 1;

            return db;
        }
    };
}

const DB_PATH = "vectors.elq";

pub fn main() !void {
    const dim = 128;
    const DB = VectorDB(dim, .Cosine);

    var db: DB = undefined;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (std.fs.cwd().access(DB_PATH, .{})) |_| {
        print("Loading...\n", .{});
        db = try DB.load(allocator, DB_PATH);
    } else |_| {
        print("Generating...\n", .{});
        db = DB.init(allocator);
    }
    defer db.deinit();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rand = prng.random();

    print("Inserting 5,000 random vectors...\n", .{});
    const empty_metadata: []const MetadataPair = &.{};
    for (0..5_000) |_| {
        var raw: @Vector(dim, f32) = undefined;
        for (0..dim) |j| {
            raw[j] = rand.float(f32) * 2.0 - 1.0;
        }
        _ = try db.add(raw, empty_metadata);
    }

    print("Saving...\n", .{});
    try db.save(DB_PATH);

    var query_raw: @Vector(dim, f32) = db.vectors.items[3999];

    for (0..dim) |j| {
        query_raw[j] += (rand.float(f32) * 0.001);
    }
    print("Searching top 10 neighbors...\n", .{});

    const top_10 = try db.search(query_raw, 10);
    defer allocator.free(top_10);

    for (top_10, 1..) |result, rank| {
        print("Rank {} → id: {}, score: {d:.6}\n", .{ rank, result.id, result.score });
    }
}

test "save and load roundtrip" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    var db = DB.init(allocator);
    defer db.deinit();

    const empty_metadata: []const MetadataPair = &.{};

    _ = try db.add(.{ 1.0, 0.0, 0.0, 0.0 }, empty_metadata);
    _ = try db.add(.{ 0.0, 1.0, 0.0, 0.0 }, empty_metadata);
    _ = try db.add(.{ 0.0, 0.0, 1.0, 0.0 }, empty_metadata);

    const test_path = "test_db.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded_db = try DB.load(allocator, test_path);
    defer loaded_db.deinit();

    try std.testing.expectEqual(3, loaded_db.ids.items.len);
    try std.testing.expectEqual(3, loaded_db.vectors.items.len);

    try std.testing.expectEqual(1, loaded_db.ids.items[0]);
    try std.testing.expectEqual(2, loaded_db.ids.items[1]);
    try std.testing.expectEqual(3, loaded_db.ids.items[2]);

    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(db.vectors.items[0][i], loaded_db.vectors.items[0][i], 1e-6);
        try std.testing.expectApproxEqAbs(db.vectors.items[1][i], loaded_db.vectors.items[1][i], 1e-6);
        try std.testing.expectApproxEqAbs(db.vectors.items[2][i], loaded_db.vectors.items[2][i], 1e-6);
    }
}

test "load with invalid magic returns error" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    const test_path = "test_invalid.elq";

    {
        var file = try std.fs.cwd().createFile(test_path, .{});
        defer file.close();
        const invalid_header = [_]u8{0} ** @sizeOf(Header);
        try file.writeAll(&invalid_header);
    }
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const result = DB.load(allocator, test_path);
    try std.testing.expectError(error.InvalidFileFormat, result);
}

test "load with dimension mismatch returns error" {
    const allocator = std.testing.allocator;

    const DB4 = VectorDB(4, .Cosine);
    var db = DB4.init(allocator);
    defer db.deinit();

    const empty_metadata: []const MetadataPair = &.{};
    _ = try db.add(.{ 1.0, 0.0, 0.0, 0.0 }, empty_metadata);

    const test_path = "test_dim_mismatch.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const DB8 = VectorDB(8, .Cosine);
    const result = DB8.load(allocator, test_path);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "search on loaded database works correctly" {
    const allocator = std.testing.allocator;
    const dim = 4;
    const DB = VectorDB(dim, .Cosine);

    var db = DB.init(allocator);
    defer db.deinit();

    const empty_metadata: []const MetadataPair = &.{};
    _ = try db.add(.{ 1.0, 0.0, 0.0, 0.0 }, empty_metadata);
    _ = try db.add(.{ 0.9, 0.1, 0.0, 0.0 }, empty_metadata);
    _ = try db.add(.{ 0.0, 1.0, 0.0, 0.0 }, empty_metadata);

    const test_path = "test_search.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded_db = try DB.load(allocator, test_path);
    defer loaded_db.deinit();

    const results = try loaded_db.search(.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(2, results.len);
    try std.testing.expectEqual(1, results[0].id);
    try std.testing.expectEqual(2, results[1].id);
}

test "metadata find_value and matches_filter" {
    const metadata: []const MetadataPair = &.{
        .{ .key = "status", .value = .{ .string = "published" } },
        .{ .key = "creator_id", .value = .{ .integer = 42 } },
    };

    try validate_metadata(metadata);

    try std.testing.expect(find_value(metadata, "status") != null);
    try std.testing.expect(find_value(metadata, "missing") == null);

    const ok_filter: []const Condition = &.{
        .{ .eq = .{ .key = "status", .value = .{ .string = "published" } } },
        .{ .eq = .{ .key = "creator_id", .value = .{ .integer = 42 } } },
    };
    try std.testing.expect(matches_filter(metadata, ok_filter));

    const bad_filter: []const Condition = &.{
        .{ .eq = .{ .key = "status", .value = .{ .string = "draft" } } },
    };
    try std.testing.expect(!matches_filter(metadata, bad_filter));
}

test "metadata rejects duplicate keys" {
    const metadata: []const MetadataPair = &.{
        .{ .key = "status", .value = .{ .string = "published" } },
        .{ .key = "status", .value = .{ .string = "draft" } },
    };
    try std.testing.expectError(error.MetadataDuplicateKey, validate_metadata(metadata));
}

test "search_filtered filters by different metadata" {
    const allocator = std.testing.allocator;
    const DB = VectorDB(4, .DotProduct);

    var db = DB.init(allocator);
    defer db.deinit();

    const id1 = try db.add(.{ 1, 0, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "published" } },
        .{ .key = "category", .value = .{ .string = "tech" } },
        .{ .key = "priority", .value = .{ .integer = 1 } },
    });

    const id2 = try db.add(.{ 0.9, 0.1, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "draft" } },
        .{ .key = "category", .value = .{ .string = "tech" } },
        .{ .key = "priority", .value = .{ .integer = 3 } },
    });

    const id3 = try db.add(.{ 0.8, 0.2, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "published" } },
        .{ .key = "category", .value = .{ .string = "science" } },
        .{ .key = "priority", .value = .{ .integer = 2 } },
    });

    const id4 = try db.add(.{ 0.7, 0.3, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "archived" } },
        .{ .key = "category", .value = .{ .string = "tech" } },
        .{ .key = "featured", .value = .{ .boolean = true } },
    });

    // Test 1: Filter by status = "published" (should return id1, id3)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "status", .value = .{ .string = "published" } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 2), results.len);
        // Results ordered by score (highest first)
        try std.testing.expectEqual(id1, results[0].id);
        try std.testing.expectEqual(id3, results[1].id);
    }

    // Test 2: Filter by category = "tech" (should return id1, id2, id4)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "category", .value = .{ .string = "tech" } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 3), results.len);
        try std.testing.expectEqual(id1, results[0].id);
        try std.testing.expectEqual(id2, results[1].id);
        try std.testing.expectEqual(id4, results[2].id);
    }

    // Test 3: Filter by status = "published" AND category = "tech" (should return id1 only)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "status", .value = .{ .string = "published" } } },
            .{ .eq = .{ .key = "category", .value = .{ .string = "tech" } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqual(id1, results[0].id);
    }

    // Test 4: Filter by priority = 1 (should return id1 only)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "priority", .value = .{ .integer = 1 } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqual(id1, results[0].id);
    }

    // Test 5: Filter by has "featured" key (should return id4 only)
    {
        const filter: []const Condition = &.{
            .{ .has = "featured" },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqual(id4, results[0].id);
    }

    // Test 6: Filter by featured = true AND category = "tech" (should return id4)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "featured", .value = .{ .boolean = true } } },
            .{ .eq = .{ .key = "category", .value = .{ .string = "tech" } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqual(id4, results[0].id);
    }

    // Test 7: Filter with no matches (should return empty)
    {
        const filter: []const Condition = &.{
            .{ .eq = .{ .key = "status", .value = .{ .string = "deleted" } } },
        };
        const results = try db.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
        defer allocator.free(results);

        try std.testing.expectEqual(@as(usize, 0), results.len);
    }
}

test "save and load preserves all metadata value types" {
    const allocator = std.testing.allocator;
    const DB = VectorDB(4, .Cosine);

    var db = DB.init(allocator);
    defer db.deinit();

    _ = try db.add(.{ 1, 0, 0, 0 }, &.{
        .{ .key = "title", .value = .{ .string = "Hello World" } },
        .{ .key = "views", .value = .{ .integer = 12345 } },
        .{ .key = "published", .value = .{ .boolean = true } },
    });
    _ = try db.add(.{ 0, 1, 0, 0 }, &.{});
    _ = try db.add(.{ 0, 0, 1, 0 }, &.{
        .{ .key = "draft", .value = .{ .boolean = false } },
        .{ .key = "score", .value = .{ .integer = -42 } },
    });

    const test_path = "test_metadata_roundtrip.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded = try DB.load(allocator, test_path);
    defer loaded.deinit();

    const meta0 = loaded.metadatas.items[0];
    try std.testing.expectEqual(3, meta0.len);
    try std.testing.expectEqualStrings("title", meta0[0].key);
    try std.testing.expectEqualStrings("Hello World", meta0[0].value.string);
    try std.testing.expectEqualStrings("views", meta0[1].key);
    try std.testing.expectEqual(12345, meta0[1].value.integer);
    try std.testing.expectEqualStrings("published", meta0[2].key);
    try std.testing.expectEqual(true, meta0[2].value.boolean);

    try std.testing.expectEqual(0, loaded.metadatas.items[1].len);

    const meta2 = loaded.metadatas.items[2];
    try std.testing.expectEqual(2, meta2.len);
    try std.testing.expectEqual(false, meta2[0].value.boolean);
    try std.testing.expectEqual(-42, meta2[1].value.integer);
}

test "search_filtered works after load" {
    const allocator = std.testing.allocator;
    const DB = VectorDB(4, .DotProduct);

    var db = DB.init(allocator);
    defer db.deinit();

    const pub_id = try db.add(.{ 1, 0, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "published" } },
    });
    _ = try db.add(.{ 0.9, 0.1, 0, 0 }, &.{
        .{ .key = "status", .value = .{ .string = "draft" } },
    });

    const test_path = "test_filter_after_load.elq";
    try db.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var loaded = try DB.load(allocator, test_path);
    defer loaded.deinit();

    const filter: []const Condition = &.{
        .{ .eq = .{ .key = "status", .value = .{ .string = "published" } } },
    };
    const results = try loaded.search_filtered(.{ 1, 0, 0, 0 }, 10, filter);
    defer allocator.free(results);

    try std.testing.expectEqual(1, results.len);
    try std.testing.expectEqual(pub_id, results[0].id);
}

test "metadata validation enforces limits" {
    try std.testing.expectError(
        error.MetadataEmptyKey,
        validate_metadata(&.{.{ .key = "", .value = .{ .boolean = true } }}),
    );

    const long_key = "a" ** (MAX_KEY_LEN + 1);
    try std.testing.expectError(
        error.MetadataKeyTooLong,
        validate_metadata(&.{.{ .key = long_key, .value = .{ .boolean = true } }}),
    );

    const long_string = "x" ** (MAX_STRING_LEN + 1);
    try std.testing.expectError(
        error.MetadataStringTooLong,
        validate_metadata(&.{.{ .key = "k", .value = .{ .string = long_string } }}),
    );
}
