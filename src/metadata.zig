const std = @import("std");

pub const MAX_METADATA_PAIRS: usize = 16;
pub const MAX_KEY_LEN: usize = 64;
pub const MAX_STRING_LEN: usize = 256;
pub const MAX_METADATA_BYTES: usize = 1024;

pub const Value = union(enum) {
    boolean: bool,
    integer: i64,
    string: []const u8,
};

pub const MetadataPair = struct {
    key: []const u8,
    value: Value,
};

pub const Condition = union(enum) {
    has: []const u8,
    eq: struct { key: []const u8, value: Value },
};

pub fn value_eq(a: Value, b: Value) bool {
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
pub fn find_value(metadata: []const MetadataPair, key: []const u8) ?Value {
    for (metadata) |pair| {
        if (std.mem.eql(u8, pair.key, key)) return pair.value;
    }
    return null;
}

pub fn validate_metadata(pairs: []const MetadataPair) !void {
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

pub fn matches_filter(metadata: []const MetadataPair, filter: []const Condition) bool {
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
