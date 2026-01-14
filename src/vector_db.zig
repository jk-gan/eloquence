const std = @import("std");

const assert = std.debug.assert;
const meta = @import("metadata.zig");
const storage = @import("storage.zig");

pub const Value = meta.Value;
pub const MetadataPair = meta.MetadataPair;
pub const Condition = meta.Condition;

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
            try meta.validate_metadata(metadata);

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
                if (!meta.matches_filter(metadata, filter)) continue;

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

            var offsets = try self.allocator.alloc(storage.OffsetEntry, count);
            defer self.allocator.free(offsets);

            for (self.metadatas.items, 0..) |metadata, i| {
                const start_offset = allocating.written().len;

                try storage.serialize_metadata(&allocating.writer, metadata);

                offsets[i] = storage.OffsetEntry{
                    .offset = @intCast(start_offset),
                    .length = @intCast(allocating.written().len - start_offset),
                    ._padding = 0,
                };
            }

            const header_size: u64 = @sizeOf(storage.Header);
            const vectors_size: u64 = count * dim * @sizeOf(f32);
            const ids_size: u64 = count * @sizeOf(u64);
            const offset_table_pos = header_size + vectors_size + ids_size;
            const blob_pos = offset_table_pos + (count * @sizeOf(storage.OffsetEntry));

            const header = storage.Header{
                .magic = storage.MAGIC_NUMBER,
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

            var header_bytes: [@sizeOf(storage.Header)]u8 = undefined;
            const header_read = try file.readAll(&header_bytes);
            if (header_read != @sizeOf(storage.Header)) {
                return error.UnexpectedEndOfFile;
            }

            const header: *const storage.Header = @ptrCast(@alignCast(&header_bytes));
            if (header.magic != storage.MAGIC_NUMBER) {
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

            const offsets = try allocator.alloc(storage.OffsetEntry, count);
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
                db.metadatas.items[i] = try storage.deserialize_metadata(&reader, db.arena.allocator());
            }

            var max_id: u64 = 0;
            for (db.ids.items) |id| max_id = @max(max_id, id);
            db.next_id = if (max_id == std.math.maxInt(u64)) max_id else max_id + 1;

            return db;
        }
    };
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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_db.elq", .{tmp.sub_path[0..]});

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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_invalid.elq", .{tmp.sub_path[0..]});

    {
        var file = try std.fs.cwd().createFile(test_path, .{});
        defer file.close();
        const invalid_header = [_]u8{0} ** @sizeOf(storage.Header);
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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_dim_mismatch.elq", .{tmp.sub_path[0..]});

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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_search.elq", .{tmp.sub_path[0..]});

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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_metadata_roundtrip.elq", .{tmp.sub_path[0..]});

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

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [128]u8 = undefined;
    const test_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/test_filter_after_load.elq", .{tmp.sub_path[0..]});

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
